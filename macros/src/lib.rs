//! Generate checked function calls and kernel adapters over compiled Enzyme derivatives.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Expr, FnArg, ItemFn, MetaNameValue, Pat, ReturnType, Token, Type};

mod function;
mod shape;

/// Give an ordinary numerical function a named, checked evaluation API.
///
/// `#[mercury::function(Rosenbrock)]` preserves the original function and creates
/// `Rosenbrock::new()`. Arguments and returns accept `f64`, `[f64; N]`, or `[[f64; C]; R]` with
/// positive integer literal dimensions. All arguments are active. When any
/// argument is an array, derivative results have fields named after arguments.
/// Scalar functions expose `gradient()` and `value_and_gradient()`; array-returning
/// functions expose `jacobian()`. Derivatives follow argument declaration order.
/// `#[function(Name, first_order)]` omits nested autodiff and the typed Hessian API.
/// By default, scalar gradients expose `jacobian().eval(...)` for the Hessian. First
/// derivative handles are also operators; third derivatives are unsupported.
/// The generated type also implements `mercury::advanced::Operator` for graph composition.
///
/// Function bodies must be deterministic, without external mutation, and valid
/// on their documented domains. Checked calls reject nonfinite inputs, values
/// and derivatives; they do not catch panics. The original Rust function remains
/// unchecked. The consuming crate needs `#![feature(autodiff)]` and Enzyme.
#[proc_macro_attribute]
pub fn function(arguments: TokenStream, item: TokenStream) -> TokenStream {
    match function::expand(arguments.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

/// Compile a slice kernel and generate its `<name>_operator(config)` constructor.
///
/// The function must have the signature
/// `fn name(config: &Config, input: &[f64], output: &mut [f64])`.
/// Configuration is inactive; all potentially differentiated values belong in
/// `input`. The `inputs` and `outputs` attributes are dimension expressions,
/// evaluated by the constructor; they may refer to its configuration argument.
/// The consuming crate needs `#![feature(autodiff)]` and the Enzyme toolchain.
///
/// ```ignore
/// #[mercury::advanced::differentiable(inputs = 2, outputs = 1)]
/// fn energy(config: &Config, input: &[f64], output: &mut [f64]) {
///     output[0] = config.scale * (input[0] * input[0] + input[1] * input[1]);
/// }
/// let operator = energy_operator(Config { scale: 0.5 });
/// ```
#[proc_macro_attribute]
pub fn differentiable(arguments: TokenStream, item: TokenStream) -> TokenStream {
    match expand(arguments.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn dimensions(arguments: proc_macro2::TokenStream) -> syn::Result<(Expr, Expr)> {
    let arguments = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse2(arguments)?;
    let mut inputs = None;
    let mut outputs = None;
    for argument in arguments {
        let slot = if argument.path.is_ident("inputs") {
            &mut inputs
        } else if argument.path.is_ident("outputs") {
            &mut outputs
        } else {
            return Err(syn::Error::new_spanned(
                argument.path,
                "expected inputs or outputs",
            ));
        };
        if slot.is_some() {
            return Err(syn::Error::new_spanned(
                argument.path,
                "duplicate dimension",
            ));
        }
        *slot = Some(argument.value);
    }
    match (inputs, outputs) {
        (Some(inputs), Some(outputs)) => Ok((inputs, outputs)),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "specify both inputs and outputs",
        )),
    }
}

fn is_float_slice(ty: &Type, mutable: bool) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return false;
    };
    let Type::Path(element) = slice.elem.as_ref() else {
        return false;
    };
    reference.mutability.is_some() == mutable && element.path.is_ident("f64")
}

fn expand(
    arguments: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let (inputs, outputs) = dimensions(arguments)?;
    let function: ItemFn = syn::parse2(item)?;
    let signature = &function.sig;
    if signature.asyncness.is_some()
        || signature.constness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || signature.variadic.is_some()
        || !signature.generics.params.is_empty()
        || signature.generics.where_clause.is_some()
    {
        return Err(syn::Error::new_spanned(
            signature,
            "expected an ordinary nongeneric Rust function",
        ));
    }
    let returns_unit = match &signature.output {
        ReturnType::Default => true,
        ReturnType::Type(_, ty) => {
            matches!(ty.as_ref(), Type::Tuple(tuple) if tuple.elems.is_empty())
        }
    };
    if signature.inputs.len() != 3 || !returns_unit {
        return Err(syn::Error::new_spanned(
            signature,
            "expected fn(config: &Config, input: &[f64], output: &mut [f64])",
        ));
    }
    let parameters: Vec<_> = signature.inputs.iter().collect();
    let FnArg::Typed(config) = parameters[0] else {
        return Err(syn::Error::new_spanned(
            signature,
            "kernel methods are unsupported",
        ));
    };
    let Pat::Ident(config_name) = config.pat.as_ref() else {
        return Err(syn::Error::new_spanned(
            &config.pat,
            "configuration must have a name",
        ));
    };
    let Type::Reference(config_reference) = config.ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &config.ty,
            "configuration must be an immutable reference",
        ));
    };
    if config_reference.mutability.is_some()
        || !matches!(
            config_reference.elem.as_ref(),
            Type::Path(_) | Type::Tuple(_)
        )
    {
        return Err(syn::Error::new_spanned(
            &config.ty,
            "use an immutable reference to an owned configuration type",
        ));
    }
    for (parameter, mutable) in [(parameters[1], false), (parameters[2], true)] {
        if !matches!(parameter, FnArg::Typed(parameter) if is_float_slice(&parameter.ty, mutable)) {
            return Err(syn::Error::new_spanned(
                parameter,
                "expected input: &[f64] or output: &mut [f64]",
            ));
        }
    }

    let name = &signature.ident;
    let visibility = &function.vis;
    let constructor = format_ident!("{name}_operator");
    let module = format_ident!("__mercury_{name}");
    let shape = format_ident!("__mercury_shape", span = proc_macro2::Span::mixed_site());
    let config_type = &config_reference.elem;
    let config_name = &config_name.ident;
    let conditions: Vec<_> = function
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg"))
        .collect();

    Ok(quote! {
        // The shared kernel ABI borrows inactive configuration, including `()`.
        #[allow(clippy::trivially_copy_pass_by_ref)]
        #function

        #(#conditions)*
        #[doc(hidden)]
        mod #module {
            #[allow(unused_imports, clippy::wildcard_imports)]
            use super::*;

            #[::std::autodiff::autodiff_forward(jvp, Const, Dual, Dual)]
            #[::std::autodiff::autodiff_reverse(vjp, Const, Duplicated, Duplicated)]
            #[allow(clippy::trivially_copy_pass_by_ref)]
            pub(super) fn primal(config: &#config_type, input: &[f64], output: &mut [f64]) {
                super::#name(config, input, output);
            }
        }

        #(#conditions)*
        #[doc = concat!("Constructs a checked differentiable operator for [`", stringify!(#name), "`].")]
        #[allow(clippy::used_underscore_binding)]
        #visibility fn #constructor(#config_name: #config_type) -> ::mercury::advanced::Kernel<#config_type> {
            let #shape = ::mercury::advanced::Shape { inputs: #inputs, outputs: #outputs };
            ::mercury::advanced::Kernel::new(#config_name, #shape, #module::primal, #module::jvp, #module::vjp)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::expand;
    use quote::quote;

    #[test]
    fn rejects_ambiguous_dimensions_and_non_slice_kernels() {
        let kernel = quote!(
            fn model(config: &(), input: &[f64], output: &mut [f64]) {}
        );
        assert!(expand(quote!(inputs = 2, inputs = 3, outputs = 1), kernel.clone()).is_err());
        assert!(expand(quote!(inputs = 2), kernel).is_err());
        let scalar = quote!(
            fn model(config: &(), input: f64, output: &mut [f64]) {}
        );
        assert!(expand(quote!(inputs = 1, outputs = 1), scalar).is_err());
    }
}
