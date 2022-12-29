use proc_macro2::TokenStream;
use quote::quote;
use anyhow::{Error, Result};

use syn::{DeriveInput, Data::Struct, Fields, DataStruct, Ident, parse2, Generics};


pub fn derive_macro_config(input: TokenStream) -> Result<TokenStream> {
    let DeriveInput {
        ident,
        data,
        generics,
        ..
    } = parse2(input).unwrap();
    
    match data {
        Struct(my_struct) => {
            config_derive::derive_macro_config_named_struct_helper(ident, my_struct, generics).into()
        },
        _ => unimplemented!(),
    }
}

pub fn derive_macro_ui(input: TokenStream) -> Result<TokenStream> {
    let DeriveInput {
        ident,
        data,
        generics,
        ..
    } = parse2(input).unwrap();
    
    match data {
        Struct(my_struct) => {
            ui_derive::derive_macro_ui_named_struct_helper(ident, my_struct, generics).into()
        },
        _ => unimplemented!(),
    }
}

/// Produces false if any field is annotated with #[no_op], true if there is no tag
/// and panics otherwise (if there is multiple tags, or if the tag is not "no_op")
fn filter_field_tag(field: &syn::Field) -> Result<bool> {
    if field.attrs.len() == 0 {
        Ok(true)
    } else if field.attrs.len() == 1 {
        if let Some(ident) = field.attrs[0].path.get_ident() {
            if ident.to_string() != "no_op".to_string() {
                return Err(Error::msg("only no_op attribute allowed"));
            }
            Ok(false)
        } else {
            Err(Error::msg("no identifier for attribute"))
        }
    } else {
        Err(Error::msg("multiple attributes detected, only 1 allowed"))
    }
}

/// Removes bounds on generic parameters, ex <T: Clone> is converted to <T>
fn strip_trait_bounds(generics: &Generics) -> Generics {
    let mut generics = generics.clone();
    for g in &mut generics.params {
        match g {
            syn::GenericParam::Type(param) => {
                param.bounds.clear();
            },
            syn::GenericParam::Lifetime(param) => {param.bounds.clear();},
            syn::GenericParam::Const(_) => {},
        }
    }
    generics
}

mod config_derive {
    use super::*;
    pub fn derive_macro_config_named_struct_helper(struct_name: Ident, s: DataStruct, generics: Generics) -> Result<TokenStream> {
        let stripped_generics = strip_trait_bounds(&generics);
        let where_clause = &generics.where_clause;
    
        match s.fields {
            Fields::Named(fields) => {
                let mut named_field_idents = Vec::new();
                for f in fields.named.iter() {
                    if filter_field_tag(f)? {
                        named_field_idents.push(&f.ident);
                    }
                }
                let tokens = quote!{
                    impl #generics Config for #struct_name #stripped_generics
                    #where_clause {
                        fn config(&self) -> String {
                            use std::collections::HashMap;
                            let mut configs = HashMap::<String, String>::new();
                            #(configs.insert(stringify!(#named_field_idents).to_string(), self.#named_field_idents.config());) *
                            ron::to_string(&configs).unwrap()
                        }
                        fn load_config(&mut self, config: &str) -> Result<()> {
                            use std::collections::HashMap;
                            let configs: HashMap::<String, String> = ron::from_str(config)?;
                            #(self.#named_field_idents.load_config(configs.get(stringify!(#named_field_idents)).unwrap())?;) *
                            Ok(())
                        }
                    }
                };
                Ok(tokens)
            }
            Fields::Unnamed(fields) => {
                let mut idx = Vec::new();
                for (f, x) in fields.unnamed.iter().zip(0..) {
                    if filter_field_tag(f)? {
                        idx.push(x);
                    }
                }
                let tokens = quote!{
                    impl #generics Config for #struct_name #stripped_generics
                    #where_clause {
                        fn config(&self) -> String {
                            use std::collections::HashMap;
                            let mut configs = HashMap::<String, String>::new();
                            #(configs.insert(stringify!(#idx).to_string(), self.#idx.config());) *
                            ron::to_string(&configs).unwrap()
                        }
                        fn load_config(&mut self, config: &str) -> Result<()> {
                            use std::collections::HashMap;
                            let configs: HashMap::<String, String> = ron::from_str(config)?;
                            #(self.#idx.load_config(configs.get(stringify!(#idx)).unwrap())?;) *
                            Ok(())
                        }
                    }
                
                };
                Ok(tokens)
            }
            Fields::Unit => {
                Ok(quote!{
                    impl #generics Config for #struct_name #stripped_generics 
                    #where_clause {
                        fn config(&self) -> String {
                            "".to_string()
                        }
                        fn load_config(&mut self, config: &str) -> Result<()> {
                            Ok(())
                        } 
                    }
                })
            }
        }
    }
    
    #[test]
    fn test_config() {
        let tokens = quote!(
            struct T {
                a: f32,
                ts: Vec<usize>
            }
        );
    
        let impl_config = derive_macro_config(tokens).unwrap();
        let expected = quote!(
            impl Config for T {
                fn config(&self) -> String {
                    use std::collections::HashMap;
                    let mut configs = HashMap::<String, String>::new();
                    configs.insert(stringify!(a).to_string(), self.a.config());
                    configs.insert(stringify!(ts).to_string(), self.ts.config());
                    ron::to_string(&configs).unwrap()
                }
                fn load_config(&mut self, config: &str) -> Result<()> {
                    use std::collections::HashMap;
                    let configs: HashMap::<String, String> = ron::from_str(config)?;
                    self.a.load_config(configs.get(stringify!(a)).unwrap())?;
                    self.ts.load_config(configs.get(stringify!(ts)).unwrap())?;
                    Ok(())
                }
            }
        );
        assert!(impl_config.to_string() == expected.to_string());
    }

    #[test]
    fn test_config_generics() {
        let tokens = quote!(
            struct T<'a, S: Clone + Config, H: Config>
            where H: Clone {
                a: &(&'a S, H)
            }
        );
    
        let impl_config = derive_macro_config(tokens).unwrap();
        let expected = quote!(
            impl<'a, S: Clone + Config, H: Config> Config for T<'a, S, H>
            where H: Clone {
                fn config(&self) -> String {
                    use std::collections::HashMap;
                    let mut configs = HashMap::<String, String>::new();
                    configs.insert(stringify!(a).to_string(), self.a.config());
                    ron::to_string(&configs).unwrap()
                }
                fn load_config(&mut self, config: &str) -> Result<()> {
                    use std::collections::HashMap;
                    let configs: HashMap::<String, String> = ron::from_str(config)?;
                    self.a.load_config(configs.get(stringify!(a)).unwrap())?;
                    Ok(())
                }
            }
        );
        assert!(impl_config.to_string() == expected.to_string());
    }

    #[test]
    fn test_config_tags() {
        let tokens = quote!(
            struct T<'a, S: Clone + Config, H: Config>
            where H: Clone {
                a: &(&'a S, H),
                #[no_op]
                b: usize,
                #[no_op]
                c: usize,
            }
        );
    
        let impl_config = derive_macro_config(tokens).unwrap();
        let expected = quote!(
            impl<'a, S: Clone + Config, H: Config> Config for T<'a, S, H>
            where H: Clone {
                fn config(&self) -> String {
                    use std::collections::HashMap;
                    let mut configs = HashMap::<String, String>::new();
                    configs.insert(stringify!(a).to_string(), self.a.config());
                    ron::to_string(&configs).unwrap()
                }
                fn load_config(&mut self, config: &str) -> Result<()> {
                    use std::collections::HashMap;
                    let configs: HashMap::<String, String> = ron::from_str(config)?;
                    self.a.load_config(configs.get(stringify!(a)).unwrap())?;
                    Ok(())
                }
            }
        );
        assert!(impl_config.to_string() == expected.to_string());
    }
    
    #[test]
    fn test_config_generics2() {
        let tokens = quote!(
            struct T <'a, S: Copy>
            where S: Clone {
                #[tag = 1]
                a: f32,
                b: f64,
                ts: Vec<usize>,
                d: &'a S
            }
        );
    
        let derive_input: DeriveInput = parse2(tokens).unwrap();
        // println!("{:#?}", derive_input);
        let generics = &derive_input.generics;
        let stripped_generics = strip_trait_bounds(&generics);
        let where_clause = &generics.where_clause;
        let new_tokens = quote!(
            impl #generics Config for T #stripped_generics
            #where_clause {
    
            }
        );
        let expected = quote!(
            impl<'a, S: Copy> Config for T<'a, S>
            where S: Clone {}
        );
        assert!( format!("{}", expected) == format!("{}", new_tokens) );
    }
}

mod ui_derive {
    use super::*;
    pub fn derive_macro_ui_named_struct_helper(struct_name: Ident, s: DataStruct, generics: Generics) -> Result<TokenStream> {
        let stripped_generics = strip_trait_bounds(&generics);
        let where_clause = &generics.where_clause;
    
        match s.fields {
            Fields::Named(fields) => {
                let mut named_field_idents = Vec::new();
                for f in fields.named.iter() {
                    if filter_field_tag(f)? {
                        named_field_idents.push(&f.ident);
                    }
                }
                let tokens = quote!{
                    impl #generics UI for #struct_name #stripped_generics
                    #where_clause {
                        fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
                            ui.vertical(|ui| {
                                #(ui.label(stringify!(#named_field_idents)); self.#named_field_idents.ui(ui);) *
                            });
                        }
                    }
                };
                Ok(tokens)
            }
            Fields::Unnamed(fields) => {
                let mut idx = Vec::new();
                for (f, x) in fields.unnamed.iter().zip(0..) {
                    if filter_field_tag(f)? {
                        idx.push(x);
                    }
                }
                let tokens = quote!{
                    impl #generics UI for #struct_name #stripped_generics
                    #where_clause {
                        fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
                            ui.vertical(|ui| {
                                #(ui.label(stringify!(#idx)); self.#idx.ui(ui);) *
                            });
                        }
                    }
                
                };
                Ok(tokens)
            }
            Fields::Unit => {
                Ok(quote!{
                    impl #generics UI for #struct_name #stripped_generics
                    #where_clause {
                        fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {}
                    }
                })
            }
        }
    }
    
    #[test]
    fn test_ui() {
        let tokens = quote!(
            struct T {
                a: f32,
                ts: Vec<usize>
            }
        );
    
        let impl_config = derive_macro_ui(tokens).unwrap();
        let expected = quote!(
            impl UI for T {
                fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
                    ui.vertical(|ui| {
                        ui.label(stringify!(a));
                        self.a.ui(ui);
                        ui.label(stringify!(ts));
                        self.ts.ui(ui);
                    });
                }
            }
        );
        assert!(impl_config.to_string() == expected.to_string());
    }

    #[test]
    fn test_ui_generics() {
        let tokens = quote!(
            struct T<'a, S: Clone + Config, H: Config>
            where H: Clone {
                a: &(&'a S, H)
            }
        );
    
        let impl_config = derive_macro_ui(tokens).unwrap();
        let expected = quote!(
            impl<'a, S: Clone + Config, H: Config> UI for T<'a, S, H>
            where H: Clone {
                fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
                    ui.vertical(|ui| {
                        ui.label(stringify!(a));
                        self.a.ui(ui);
                    });
                }
            }
        );
        assert!(impl_config.to_string() == expected.to_string());
    }

    #[test]
    fn test_ui_tags() {
        let tokens = quote!(
            struct T<'a, S: Clone + Config, H: Config>
            where H: Clone {
                a: &(&'a S, H),
                #[no_op]
                b: usize,
                #[no_op]
                c: usize,
            }
        );
    
        let impl_config = derive_macro_ui(tokens).unwrap();
        let expected = quote!(
            impl<'a, S: Clone + Config, H: Config> UI for T<'a, S, H>
            where H: Clone {
                fn ui(&mut self, ui: &mut bevy_egui::egui::Ui) {
                    ui.vertical(|ui| {
                        ui.label(stringify!(a));
                        self.a.ui(ui);
                    });
                }
            }
        );
        assert!(impl_config.to_string() == expected.to_string());
    }
}