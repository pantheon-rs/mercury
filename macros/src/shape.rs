//! Fixed scalar, vector and matrix layouts at the typed boundary.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Lit, Type};

pub struct Shape {
    dimensions: Vec<usize>,
}

impl Shape {
    pub fn parse(ty: &Type) -> syn::Result<Self> {
        let mut element = ty;
        let mut dimensions = Vec::new();
        while let Type::Array(array) = element {
            let Expr::Lit(length) = &array.len else {
                return Err(invalid(ty));
            };
            let Lit::Int(length) = &length.lit else {
                return Err(invalid(ty));
            };
            let size = length.base10_parse::<usize>()?;
            if size == 0 || dimensions.len() == 2 {
                return Err(invalid(ty));
            }
            dimensions.push(size);
            element = &array.elem;
        }
        if !matches!(element, Type::Path(path) if path.qself.is_none() && path.path.is_ident("f64"))
        {
            return Err(invalid(ty));
        }
        dimensions
            .iter()
            .try_fold(1usize, |product, &size| product.checked_mul(size))
            .ok_or_else(|| invalid(ty))?;
        Ok(Self { dimensions })
    }

    pub fn size(&self) -> usize {
        self.dimensions.iter().product()
    }
    pub fn scalar(&self) -> bool {
        self.dimensions.is_empty()
    }

    pub fn flatten(&self, value: TokenStream) -> Vec<TokenStream> {
        let mut entries = vec![value];
        for &size in &self.dimensions {
            entries = entries
                .into_iter()
                .flat_map(|entry| (0..size).map(move |i| quote!(#entry[#i])))
                .collect();
        }
        entries
    }

    pub fn restore(&self, flat: &TokenStream, offset: usize) -> TokenStream {
        fn array(dimensions: &[usize], flat: &TokenStream, offset: usize) -> TokenStream {
            let Some((&size, tail)) = dimensions.split_first() else {
                return quote!(#flat[#offset]);
            };
            let stride: usize = tail.iter().product();
            let entries = (0..size).map(|i| array(tail, flat, offset + i * stride));
            quote!([#(#entries),*])
        }
        array(&self.dimensions, flat, offset)
    }
}

fn invalid(ty: &Type) -> syn::Error {
    syn::Error::new_spanned(
        ty,
        "expected f64, [f64; N], or [[f64; C]; R] with positive integer literal dimensions",
    )
}
