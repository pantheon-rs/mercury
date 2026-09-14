//! Typed function calls over the same compiled entry points used by plans.

use crate::shape::Shape;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::{FnArg, Ident, ItemFn, Pat, ReturnType, Token};

struct Options {
    name: Ident,
    second_order: bool,
}

impl Parse for Options {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let name = input.parse()?;
        let second_order = if input.is_empty() {
            true
        } else {
            input.parse::<Token![,]>()?;
            let option: Ident = input.parse()?;
            if option != "first_order" {
                return Err(syn::Error::new_spanned(option, "expected first_order"));
            }
            false
        };
        Ok(Self { name, second_order })
    }
}

pub fn expand(arguments: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let Options {
        name: type_name,
        second_order,
    } = syn::parse2(arguments)?;
    let derivative_order = if second_order { 2u8 } else { 1u8 };
    let handle_order = derivative_order - 1;
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
            "expected an ordinary nongeneric function with at least one numerical argument",
        ));
    }
    let mut names = Vec::new();
    let mut types = Vec::new();
    let mut shapes = Vec::new();
    for argument in &signature.inputs {
        if let FnArg::Typed(argument) = argument
            && let Pat::Ident(name) = argument.pat.as_ref()
            && name.by_ref.is_none()
            && name.subpat.is_none()
        {
            shapes.push(Shape::parse(&argument.ty)?);
            types.push(&argument.ty);
            names.push(&name.ident);
            continue;
        }
        return Err(syn::Error::new_spanned(
            argument,
            "expected a named scalar, vector, or matrix argument",
        ));
    }
    let ReturnType::Type(_, result_type) = &signature.output else {
        return Err(syn::Error::new_spanned(
            &signature.output,
            "expected a scalar, vector, or matrix return",
        ));
    };
    let result_shape = Shape::parse(result_type)?;
    let outputs = result_shape.size();
    let scalar = result_shape.scalar();
    let inputs = shapes
        .iter()
        .try_fold(0usize, |sum, shape| sum.checked_add(shape.size()))
        .ok_or_else(|| syn::Error::new_spanned(signature, "input dimensions overflow"))?;
    inputs
        .checked_mul(outputs)
        .ok_or_else(|| syn::Error::new_spanned(signature, "Jacobian dimensions overflow"))?;
    let mut offset = 0;
    let calls: Vec<_> = shapes
        .iter()
        .map(|shape| {
            let call = shape.restore(&quote!(input), offset);
            offset += shape.size();
            call
        })
        .collect();
    let packed: Vec<_> = names
        .iter()
        .zip(&shapes)
        .flat_map(|(name, shape)| shape.flatten(quote!(#name)))
        .collect();
    let structured = shapes.iter().any(|shape| !shape.scalar());
    let gradient_type = if structured {
        quote!(GradientValue)
    } else {
        quote!([f64; #inputs])
    };
    let name = &signature.ident;
    let visibility = &function.vis;
    let module = format_ident!("__mercury_{}", name.unraw());
    let public_gradient_type = if structured {
        quote!(#module::GradientValue)
    } else {
        quote!([f64; #inputs])
    };
    let conditions: Vec<_> = function
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("cfg"))
        .collect();
    let result_value = result_shape.restore(&quote!(value), 0);
    let call = quote!(super::#name(#(#calls),*));
    let entries = result_shape.flatten(quote!(value));
    let rows = 0..outputs;
    let write_value = quote! { let value = #call; #(output[#rows] = #entries;)* };
    let mut offset = 0;
    let gradient_fields: Vec<_> = shapes
        .iter()
        .map(|shape| {
            let field = shape.restore(&quote!(gradient), offset);
            offset += shape.size();
            field
        })
        .collect();
    let gradient_value = if structured {
        quote!(GradientValue { #(#names: #gradient_fields),* })
    } else {
        quote!(gradient)
    };
    let mut offset = 0;
    let jacobian_fields: Vec<_> = shapes
        .iter()
        .map(|shape| {
            let rows = (0..outputs).map(|row| shape.restore(&quote!(matrix[#row]), offset));
            let field = quote!([#(#rows),*]);
            offset += shape.size();
            field
        })
        .collect();
    let jacobian_value = if structured {
        quote!(JacobianValue { #(#names: #jacobian_fields),* })
    } else {
        quote!(matrix)
    };
    let jacobian_type = if structured {
        quote!(JacobianValue)
    } else {
        quote!([[f64; #inputs]; #outputs])
    };
    let value_fields = if structured {
        let value_name = if scalar {
            format_ident!("GradientValue")
        } else {
            format_ident!("JacobianValue")
        };
        let field_types: Vec<_> = types
            .iter()
            .map(|ty| {
                if scalar {
                    quote!(#ty)
                } else {
                    quote!([#ty; #outputs])
                }
            })
            .collect();
        quote! {
            /// Derivatives grouped by the original argument names and shapes.
            #[derive(Clone, Copy, Debug, PartialEq)]
            pub struct #value_name {
                #(#[doc = concat!("Partial derivatives with respect to ", stringify!(#names), ".")]
                  pub #names: #field_types,)*
            }
        }
    } else {
        quote!()
    };
    let jacobian_sweep = if outputs < inputs {
        quote! {
            for row in 0..#outputs {
                let mut seed = [0.0; #outputs];
                seed[row] = 1.0;
                let mut value = [f64::NAN; #outputs];
                vjp(&(), &input, &mut matrix[row], &mut value, &mut seed);
                check_finite("function output", &value)?;
                check_finite("Jacobian", &matrix[row])?;
            }
        }
    } else {
        quote! {
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
        }
    };
    let (derivative_name, derivative_type, derivative_eval, function_methods, derivative_helper) =
        if scalar {
            (
                format_ident!("Gradient"),
                gradient_type.clone(),
                quote!(Ok(self::value_and_gradient([#(#packed),*])?.1)),
                quote! {
                    /// Select the compiled gradient. No evaluation occurs here.
                    pub const fn gradient(&self) -> #module::Gradient {
                        #module::Gradient
                    }

                    /// Evaluate the value and gradient in one combined reverse call.
                    ///
                    /// # Errors
                    /// Returns an error for nonfinite inputs, value or derivatives.
                    pub fn value_and_gradient(&self, #(#names: #types),*)
                        -> ::mercury::Result<(f64, #public_gradient_type)>
                    {
                        #module::value_and_gradient([#(#packed),*])
                    }
                },
                quote! {
                    pub(super) fn value_and_gradient(input: [f64; #inputs])
                        -> ::mercury::Result<(f64, #gradient_type)>
                    {
                        check_finite("input", &input)?;
                        let mut value = [f64::NAN];
                        let mut gradient = [0.0; #inputs];
                        let mut seed = [1.0];
                        vjp(&(), &input, &mut gradient, &mut value, &mut seed);
                        check_finite("function output", &value)?;
                        check_finite("gradient", &gradient)?;
                        Ok((value[0], #gradient_value))
                    }
                },
            )
        } else {
            (
                format_ident!("Jacobian"),
                jacobian_type.clone(),
                quote!(self::jacobian([#(#packed),*])),
                quote! {
                    /// Select the compiled Jacobian. Rows are outputs; columns are arguments.
                    pub const fn jacobian(&self) -> #module::Jacobian {
                        #module::Jacobian
                    }
                },
                quote! {
                    fn jacobian(input: [f64; #inputs])
                        -> ::mercury::Result<#jacobian_type>
                    {
                        check_finite("input", &input)?;
                        let mut matrix = [[0.0; #inputs]; #outputs];
                        #jacobian_sweep
                        Ok(#jacobian_value)
                    }
                },
            )
        };

    let second_derivative = if scalar && second_order {
        quote! {
            impl Gradient {
                /// Select the Jacobian of the gradient: the Hessian.
                pub const fn jacobian(&self) -> Hessian { Hessian }
            }

            /// A scalar function's second derivatives in flattened argument order.
            #[derive(Clone, Copy, Debug)]
            pub struct Hessian;

            #[allow(clippy::unused_self, clippy::used_underscore_binding)]
            impl Hessian {
                /// Evaluate the Hessian; rows and columns follow argument order.
                ///
                /// # Errors
                /// Returns an error for nonfinite inputs, values or derivatives.
                pub fn eval(&self, #(#names: #types),*) -> ::mercury::Result<[[f64; #inputs]; #inputs]> {
                    self::hessian([#(#packed),*])
                }
            }

            fn hessian(input: [f64; #inputs]) -> ::mercury::Result<[[f64; #inputs]; #inputs]> {
                self::evaluate(input)?;
                let mut matrix = [[0.0; #inputs]; #inputs];
                for column in 0..#inputs {
                    let mut seed = [0.0; #inputs];
                    seed[column] = 1.0;
                    let mut product = [0.0; #inputs];
                    curvature(&(), &input, &[1.0], &seed, &mut product);
                    check_finite("Hessian", &product)?;
                    for row in 0..#inputs { matrix[row][column] = product[row]; }
                }
                Ok(matrix)
            }
        }
    } else {
        quote!()
    };

    let weight_indices = 0..outputs;
    let curvature_functions = if second_order {
        quote! {
                #[::std::autodiff::autodiff_forward(curvature_jvp, Const, Dual, Const, Dual)]
                #[allow(clippy::trivially_copy_pass_by_ref)]
                fn weighted_gradient(config: &(), input: &[f64], weights: &[f64], output: &mut [f64]) {
                    let mut value = [0.0; #outputs];
                    // A slice copy here fails nested Enzyme type inference on the pinned compiler.
                    let mut seed = [#(weights[#weight_indices]),*];
                    output.fill(0.0);
                    vjp(config, input, output, &mut value, &mut seed);
                }

                #[allow(clippy::trivially_copy_pass_by_ref)]
                fn curvature(config: &(), input: &[f64], weights: &[f64], direction: &[f64], output: &mut [f64]) {
                    let mut gradient = [0.0; #inputs];
                    curvature_jvp(config, input, direction, weights, &mut gradient, output);
                }

        }
    } else {
        quote!()
    };
    let attach_curvature = if second_order {
        quote!(.with_curvature(curvature))
    } else {
        quote!()
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
            pub fn eval(&self, #(#names: #types),*) -> ::mercury::Result<#result_type> {
                let value = #module::evaluate([#(#packed),*])?;
                Ok(#result_value)
            }

            #function_methods
        }

        #(#conditions)*
        impl ::mercury::advanced::Operator for #type_name {
            fn derivative_order(&self) -> u8 { #derivative_order }
            fn shape(&self) -> ::mercury::advanced::Shape {
                ::mercury::advanced::Operator::shape(&#module::KERNEL)
            }

            fn workspace(&self) -> Box<dyn ::mercury::advanced::OperatorWorkspace + '_> {
                ::mercury::advanced::Operator::workspace(&#module::KERNEL)
            }
        }

        #(#conditions)*
        #[doc(hidden)]
        #visibility mod #module {
            use ::mercury::__private::check_finite;

            pub(super) static KERNEL: ::mercury::advanced::Kernel<()> = ::mercury::advanced::Kernel::new(
                (), ::mercury::advanced::Shape { inputs: #inputs, outputs: #outputs }, primal, jvp, vjp,
            )#attach_curvature;

            #[::std::autodiff::autodiff_forward(jvp, Const, Dual, Dual)]
            #[::std::autodiff::autodiff_reverse(vjp, Const, Duplicated, Duplicated)]
            #[allow(clippy::trivially_copy_pass_by_ref)]
            fn primal(_config: &(), input: &[f64], output: &mut [f64]) {
                #write_value
            }

            #curvature_functions

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
                pub fn eval(&self, #(#names: #types),*) -> ::mercury::Result<#derivative_type> {
                    #derivative_eval
                }
            }

            impl ::mercury::advanced::Operator for #derivative_name {
                fn derivative_order(&self) -> u8 { #handle_order }
                fn shape(&self) -> ::mercury::advanced::Shape {
                    ::mercury::advanced::Shape { inputs: #inputs, outputs: #inputs * #outputs }
                }
                fn workspace(&self) -> Box<dyn ::mercury::advanced::OperatorWorkspace + '_> {
                    ::mercury::__private::derivative_workspace(&KERNEL, #scalar)
                }
            }

            #second_derivative
            #value_fields
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
    #[test]
    fn first_order_expansion_omits_nested_autodiff() {
        let body = quote!(
            fn square(x: f64) -> f64 {
                x * x
            }
        );
        let first = expand(quote!(Square, first_order), body.clone())
            .unwrap()
            .to_string();
        assert!(!first.contains("curvature_jvp"));
        assert!(!first.contains("struct Hessian"));
        let second = expand(quote!(Square), body.clone()).unwrap().to_string();
        assert!(second.contains("curvature_jvp"));
        assert!(expand(quote!(Square, unknown), body.clone()).is_err());
        assert!(expand(quote!(Square, first_order, first_order), body).is_err());
    }
}
