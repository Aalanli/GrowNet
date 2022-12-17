use proc_macro2::TokenStream as TokenStream2;

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
fn test_for_each_field() {
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
    let DeriveInput {
        ident,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);
    
    match data {
        Struct(my_struct) => {
            named_struct_impl(ident, my_struct).into()
        },
        _ => unimplemented!(),
    }
}

fn named_struct_impl(struct_name: Ident, s: DataStruct) -> TokenStream2 {
    match s.fields {
        Fields::Named(fields) => {
            let named_field_idents: Vec<_> = fields.named.iter().map(|x| {&x.ident}).collect();
            quote!{
                impl Config for #struct_name {
                    fn config(&self) -> String {
                        use std::collections::HashMap;
                        let mut configs = HashMap::<String, String>::new();
                        {#(configs.insert(stringify!(#named_field_idents).to_string(), self.#named_field_idents.config());) *}
                        ron::to_string(&configs).unwrap()
                    }
                    fn load_config(&mut self, config: &str) -> Result<()> {
                        use std::collections::HashMap;
                        let configs: HashMap::<String, String> = ron::from_str(config)?;
                        {#(self.#named_field_idents.load_config(configs.get(stringify!(#named_field_idents)).unwrap())?;) *}
                        Ok(())
                    }
                }
            }
        }
        Fields::Unnamed(fields) => {
            let unnamed_count = fields.unnamed.iter().count();
            let idx: Vec<_> = (0..unnamed_count).collect();
            quote!{
                impl Config for #struct_name {
                    fn config(&self) -> String {
                        use std::collections::HashMap;
                        let mut configs = HashMap::<String, String>::new();
                        {#(configs.insert(stringify!(#idx).to_string(), self.#idx.config());) *}
                        ron::to_string(&configs).unwrap()
                    }
                    fn load_config(&mut self, config: &str) -> Result<()> {
                        use std::collections::HashMap;
                        let configs: HashMap::<String, String> = ron::from_str(config)?;
                        {#(self.#idx.load_config(configs.get(stringify!(#idx)).unwrap())?;) *}
                        Ok(())
                    }
                }
            
            }
        }
        Fields::Unit => {
            quote!{
                impl Config for #struct_name {
                    fn config(&self) -> String {
                        "".to_string()
                    }
                    fn load_config(&mut self, config: &str) -> Result<()> {
                        Ok(())
                    } 
                }
            }
        }
    }
}
