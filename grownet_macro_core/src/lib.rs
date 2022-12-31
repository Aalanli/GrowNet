use proc_macro2::TokenStream;
use quote::quote;
use anyhow::{Error, Result};

use syn::{DeriveInput, Data::Struct, Fields, Ident, parse2, Generics, WhereClause, Attribute};


pub fn derive_macro_config(input: TokenStream) -> Result<TokenStream> {
    let derive_input = parse2::<DeriveInput>(input)?;
    let data_fields = BasicStructFields::new(derive_input)?;
    let struct_name = data_fields.name;
    let generics = data_fields.generics;
    let where_clause = data_fields.where_clause;
    let stripped_generics = data_fields.stripped_generics;
    let named_field_idents: Vec<_> = data_fields.fields.iter().map(|x| x.0.clone()).collect();
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

pub fn derive_macro_ui(input: TokenStream) -> Result<TokenStream> {
    let derive_input = parse2::<DeriveInput>(input)?;
    let data_fields = BasicStructFields::new(derive_input)?;
    let struct_name = data_fields.name;
    let generics = data_fields.generics;
    let where_clause = data_fields.where_clause;
    let stripped_generics = data_fields.stripped_generics;
    let named_field_idents: Vec<_> = data_fields.fields.iter().map(|x| x.0.clone()).collect();

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

/// Filters the no_op attribute from the list of fields
fn filter_tag_no_op(field: &syn::Field) -> Vec<Attribute> {
    field.attrs.iter().filter(|x| {
        if let Some(ident) = x.path.get_ident() {
            if ident.to_string() != "no_op".to_string() {
                return true;
            }
        }
        return false;
    }).map(|x| x.clone()).collect()
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

struct BasicStructFields {
    generics: Generics,
    stripped_generics: Generics,
    where_clause: Option<WhereClause>,
    name: Ident,
    fields: Vec<(TokenStream, Vec<Attribute>)>,
    //field_type: Vec<Type>
}

impl BasicStructFields {
    fn new(derive: DeriveInput) -> Result<Self> {
        let stripped_generics = strip_trait_bounds(&derive.generics);
        let where_clause = derive.generics.where_clause.clone();
        let (fields, _field_type) = if let Struct(data) = derive.data {
            match data.fields {
                Fields::Named(fields) => {
                    let mut named_field_idents = Vec::new();
                    let mut field_type = Vec::new();
                    for f in fields.named.iter() {
                        let filtered_attrs = filter_tag_no_op(f);
                        if filtered_attrs.len() == f.attrs.len() {
                            if let Some(id) = &f.ident {
                                named_field_idents.push((quote!(#id), filtered_attrs));
                                field_type.push(f.ty.clone());
                            }
                        }
                    }
                    (named_field_idents, field_type)
                }
                Fields::Unnamed(fields) => {
                    let mut idx = Vec::new();
                    let mut types = Vec::new();
                    for (f, x) in fields.unnamed.iter().zip(0..) {
                        let filtered_attrs = filter_tag_no_op(f);
                        if filtered_attrs.len() == f.attrs.len() {
                            idx.push(((x.to_string()).parse().unwrap(), filtered_attrs));
                            types.push(f.ty.clone());
                        }
                    }
                    (idx, types)
                }
                Fields::Unit => {
                    (Vec::new(), Vec::new())
                }
            }
        } else {
            return Err(Error::msg("expected data struct"));
        };
        Ok(Self { 
            generics: derive.generics, 
            stripped_generics, where_clause, 
            name: derive.ident, 
            fields, 
            //field_type 
        })
    }
}


#[cfg(test)]
mod config_derive {
    use super::*;
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
        //println!("{}", impl_config.to_string());
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
                #[tag = "1 + 2"]
                a: f32,
            }
        );
    
        let derive_input: DeriveInput = parse2(tokens).unwrap();
        println!("{:#?}", derive_input);
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

    #[test]
    fn test_struct_attr() {
        let tokens = quote!(
            #[BuildTrait(VALUE)]
            struct T {
                u: usize
            }
        );

        let derive_input: DeriveInput = parse2(tokens).unwrap();
        println!("{:#?}", derive_input);
    }
}

#[cfg(test)]
mod ui_derive {
    use super::*;
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