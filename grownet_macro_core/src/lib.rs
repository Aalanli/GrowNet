use proc_macro2::TokenStream;
use quote::quote;

use syn::{DeriveInput, Data::Struct, Fields, DataStruct, Ident, parse2};


pub fn derive_macro_config(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        data,
        ..
    } = parse2(input).unwrap();
    
    match data {
        Struct(my_struct) => {
            derive_macro_config_named_struct_helper(ident, my_struct).into()
        },
        _ => unimplemented!(),
    }
}

fn derive_macro_config_named_struct_helper(struct_name: Ident, s: DataStruct) -> TokenStream {
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
