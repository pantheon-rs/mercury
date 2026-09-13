//! Typed function calls over the same compiled entry points used by plans.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::{Expr, FnArg, Ident, ItemFn, Lit, Pat, ReturnType, Type};

fn is_float(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.is_ident("f64"))
}

fn output_shape(output: &ReturnType) -> syn::Result<(usize, bool)> {
    if let ReturnType::Type(_, ty) = output {
        if is_float(ty) {
            return Ok((1, true));
        }
        if let Type::Array(array) = ty.as_ref()
            && let Expr::Lit(length) = &array.len
            && let Lit::Int(length) = &length.lit
        {
            let size = length.base10_parse::<usize>()?;
            if is_float(&array.elem) && size > 0 {
                return Ok((size, false));
            }
        }
    }
    Err(syn::Error::new_spanned(
        output,
        "return f64 or [f64; N], where N is a positive integer literal",
    ))
}

pub fn expand(arguments: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let type_name: Ident = syn::parse2(arguments)?;
    let function: ItemFn = syn::parse2(item)?;
    let signature = &function.sig;
    if signature.asyncness.is_some()
        || signature.constness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || signature.variadic.is_some()
        || !signature.generics.params.is_empty()
        || signature.generics.where_clause.is_some()
        || signature.inputs.is_empty()
    {
        return Err(syn::Error::new_spanned(
            signature,
            "expected an ordinary nongeneric function with at least one f64 argument",
        ));
    }
    let mut names = Vec::new();
    for argument in &signature.inputs {
        if let FnArg::Typed(argument) = argument
            && let Pat::Ident(name) = argument.pat.as_ref()
            && is_float(&argument.ty)
            && name.by_ref.is_none()
            && name.subpat.is_none()
        {
            names.push(&name.ident);
            continue;
        }
        return Err(syn::Error::new_spanned(
            argument,
            "expected a named f64 argument",
        ));
    }
    let (outputs, scalar) = output_shape(&signature.output)?;
    let inputs = names.len();
    let indices = 0..inputs;
    let name = &signature.ident;
    let visibility = &function.vis;
    let module = format_ident!("__mercury_{}", name.unraw());
    let conditions: Vec<_> = function
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg"))
        .collect();
    let result_type = if scalar {
        quote!(f64)
    } else {
        quote!([f64; #outputs])
    };
    let result_value = if scalar {
        quote!(value[0])
    } else {
        quote!(value)
    };
    let call = quote!(super::#name(#(input[#indices]),*));
    let write_value = if scalar {
        quote!(output[0] = #call;)
    } else {
        // Explicit writes keep the adapter's output contract visible to Enzyme.
        let rows = 0..outputs;
        quote! {
            let value = #call;
            #(output[#rows] = value[#rows];)*
        }
    };
    let (derivative_name, derivative_type, derivative_eval, function_methods, derivative_helper) =
        if scalar {
            (
                format_ident!("Gradient"),
                quote!([f64; #inputs]),
                quote!(Ok(self::value_and_gradient([#(#names),*])?.1)),
                quote! {
                    /// Select the compiled gradient. No evaluation occurs here.
                    pub const fn gradient(&self) -> #module::Gradient {
                        #module::Gradient
                    }

                    /// Evaluate the value and gradient in one combined reverse call.
                    ///
                    /// # Errors
                    /// Returns an error for nonfinite inputs, value or derivatives.
                    pub fn value_and_gradient(&self, #(#names: f64),*)
                        -> ::mercury::Result<(f64, [f64; #inputs])>
                    {
                        #module::value_and_gradient([#(#names),*])
                    }
                },
                quote! {
                    pub(super) fn value_and_gradient(input: [f64; #inputs])
                        -> ::mercury::Result<(f64, [f64; #inputs])>
                    {
                        check_finite("input", &input)?;
                        let mut value = [f64::NAN];
                        let mut gradient = [0.0; #inputs];
                        let mut seed = [1.0];
                        vjp(&(), &input, &mut gradient, &mut value, &mut seed);
                        check_finite("function output", &value)?;
                        check_finite("gradient", &gradient)?;
                        Ok((value[0], gradient))
                    }
                },
            )
        } else {
            (
                format_ident!("Jacobian"),
                quote!([[f64; #inputs]; #outputs]),
                quote!(self::jacobian([#(#names),*])),
                quote! {
                    /// Select the compiled Jacobian. Rows are outputs; columns are arguments.
                    pub const fn jacobian(&self) -> #module::Jacobian {
                        #module::Jacobian
                    }
                },
                quote! {
                    fn jacobian(input: [f64; #inputs])
                        -> ::mercury::Result<[[f64; #inputs]; #outputs]>
                    {
                        check_finite("input", &input)?;
                        let mut matrix = [[0.0; #inputs]; #outputs];
                        for column in 0..#inputs {
                            let mut seed = [0.0; #inputs];
                            seed[column] = 1.0;
                            let mut value = [f64::NAN; #outputs];
                            let mut tangent = [0.0; #outputs];
                            jvp(&(), &input, &seed, &mut value, &mut tangent);
                            check_finite("function output", &value)?;
                            check_finite("Jacobian", &tangent)?;
                            for row in 0..#outputs {
                                matrix[row][column] = tangent[row];
                            }
                        }
                        Ok(matrix)
                    }
                },
            )
        };

    Ok(quote! {
        #function

        #(#conditions)*
        #[doc = concat!("Checked compiled function for [`", stringify!(#name), "`].")]
        #[derive(Clone, Copy, Debug, Default)]
        #visibility struct #type_name;

        #(#conditions)*
        #[allow(clippy::unused_self, clippy::used_underscore_binding)]
        impl #type_name {
            /// Construct a stateless function handle; compilation happens during the build.
            pub const fn new() -> Self { Self }

            /// Evaluate the function at the supplied arguments.
            ///
            /// # Errors
            /// Returns an error for nonfinite inputs or outputs.
            pub fn eval(&self, #(#names: f64),*) -> ::mercury::Result<#result_type> {
                let value = #module::evaluate([#(#names),*])?;
                Ok(#result_value)
            }

            #function_methods
        }

        #(#conditions)*
        impl ::mercury::Operator for #type_name {
            fn shape(&self) -> ::mercury::Shape {
                ::mercury::Operator::shape(&#module::KERNEL)
            }

            fn workspace(&self) -> Box<dyn ::mercury::OperatorWorkspace + '_> {
                ::mercury::Operator::workspace(&#module::KERNEL)
            }
        }

        #(#conditions)*
        #[doc(hidden)]
        #visibility mod #module {
            use ::mercury::__private::check_finite;

            pub(super) static KERNEL: ::mercury::Kernel<()> = ::mercury::Kernel::new(
                (), ::mercury::Shape { inputs: #inputs, outputs: #outputs }, primal, jvp, vjp,
            );

            #[::std::autodiff::autodiff_forward(jvp, Const, Dual, Dual)]
            #[::std::autodiff::autodiff_reverse(vjp, Const, Duplicated, Duplicated)]
            #[allow(clippy::trivially_copy_pass_by_ref)]
            fn primal(_config: &(), input: &[f64], output: &mut [f64]) {
                #write_value
            }

            pub(super) fn evaluate(input: [f64; #inputs])
                -> ::mercury::Result<[f64; #outputs]>
            {
                check_finite("input", &input)?;
                let mut value = [f64::NAN; #outputs];
                primal(&(), &input, &mut value);
                check_finite("function output", &value)?;
                Ok(value)
            }

            /// A stateless derivative handle; each evaluation supplies a fresh point.
            #[derive(Clone, Copy, Debug)]
            pub struct #derivative_name;

            #[allow(clippy::unused_self, clippy::used_underscore_binding)]
            impl #derivative_name {
                /// Evaluate first derivatives in argument declaration order.
                ///
                /// # Errors
                /// Returns an error for nonfinite inputs, values or derivatives.
                pub fn eval(&self, #(#names: f64),*) -> ::mercury::Result<#derivative_type> {
                    #derivative_eval
                }
            }

            #derivative_helper
        }
    })
}

#[cfg(test)]
mod tests {
    use super::expand;
    use quote::quote;

    #[test]
    fn rejects_unsupported_signatures() {
        for function in [
            quote!(
                fn f(x: &[f64]) -> f64 {
                    x[0]
                }
            ),
            quote!(
                fn f<T>(x: f64) -> f64 {
                    x
                }
            ),
            quote!(
                fn f(x: f64) -> [f64; 0] {
                    []
                }
            ),
            quote!(
                fn f(x: f64) -> [f64; SIZE] {
                    [x; SIZE]
                }
            ),
            quote!(
                fn f(x: f64) -> (f64, f64) {
                    (x, x)
                }
            ),
            quote!(
                fn f() -> f64 {
                    1.0
                }
            ),
            quote!(
                async fn f(x: f64) -> f64 {
                    x
                }
            ),
        ] {
            assert!(expand(quote!(Function), function).is_err());
        }
        assert!(
            expand(
                quote!(),
                quote!(
                    fn f(x: f64) -> f64 {
                        x
                    }
                )
            )
            .is_err()
        );
    }
}
