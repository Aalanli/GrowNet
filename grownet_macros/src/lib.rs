use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input, Data::Struct, Fields, DataStruct, Ident};


#[proc_macro_derive(Test)]
pub fn derive_test_macro(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);

    if let Struct(my_struct) = data {
        match my_struct.fields {
            Fields::Named(fields) => {
                let field_idents = fields.named.iter().map(|x| &x.ident);
                quote!(
                    impl Test for #ident {
                        fn call(&self) {
                            {#(self.#field_idents.call(); println!("{}", stringify!(#field_idents));) *}
                        }
                    }
                ).into()
            }
            _ => unimplemented!()
        }
    } else {
        unimplemented!()
    }
}


#[test]
pub fn test_for_each_field() {
    struct S {
        a: f32,
        b: f32,
        c: f32
    }
    let a = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let it = a.iter();
    let p = "a".to_string();
    let s = quote!(
        fn t(s: S) {
            {#(println!("{}", s.#a())); *}
        }
        fn h(s: S) {
            {#(println!("{}", s.#it())); *}
        }
        
    );

    println!("{}", s);
}


#[proc_macro_derive(Config)]
pub fn derive_macro_config(input: TokenStream) -> TokenStream {
    macros::derive_macro_config(input.into()).into()
}
