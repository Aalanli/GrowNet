use anyhow::{Error, Result, Context};
use proc_macro2::{TokenStream, Span};
use quote::quote;

use syn::{parse2, Attribute, Data::Struct, DeriveInput, Fields, Generics, Ident, WhereClause};
mod old;

/// two kinds of of attributes, one is #[flat(skip)], which does not flatten
/// that argument, the other is #[flat(exclude)], which does not add that field
/// into the world.
pub fn derive_flatten(input: TokenStream) -> Result<TokenStream> {
    let derive = parse2::<DeriveInput>(input)?;
    let extracted_fields = extract_fields(&derive).context("failed to extract struct fields")?;
    let SimpleDataStruct { 
        generics, 
        stripped_generics, 
        where_clause, 
        name: struct_name, 
    } = SimpleDataStruct::new(derive)?;

    let new_fields = match extracted_fields {
        SimpleStructFields::Named(fields) => {
            let new_fields: Result<Vec<_>> = fields.iter().map(|(id, attrs)| {
                let attr_arg = compute_attributes(&attrs)?;
                Ok((id.to_string(), attr_arg))
            }).collect();
            new_fields?
        }
        SimpleStructFields::Unnamed(fields) => {
            let new_fields: Result<Vec<_>> = fields.iter().enumerate().map(|(id, attrs)| {
                let attr_arg = compute_attributes(&attrs)?;
                Ok((id.to_string(), attr_arg))
            }).collect();
            new_fields?
        }
        SimpleStructFields::Unit => {
            Vec::new()
        }
    };

    let mut commands = Vec::new();
    for (name, opt) in new_fields[..new_fields.len() - 1].iter() {
        let new_name = "/".to_string() + name;
        let field_name = Ident::new(name, Span::call_site());
        let code = match opt {
            FlatAttrOptions::Include => {
                quote!(
                    self.#field_name.flatten(path.clone() + #new_name, world);
                )
            }
            FlatAttrOptions::Skip => {
                quote!(
                    world.push(path, self);
                )
            },
            FlatAttrOptions::Exclude => {
                quote!()
            },
        };
        commands.push(code);
    }
    if new_fields.len() > 0 {
        let (name, opt) = &new_fields[new_fields.len() - 1];
        let new_name = "/".to_string() + name;
        let field_name = Ident::new(name, Span::call_site());
        let code = match opt {
            FlatAttrOptions::Include => {
                quote!(
                    self.#field_name.flatten(path + #new_name, world);
                )
            },
            FlatAttrOptions::Skip => {
                quote!(
                    world.push(path, self);
                )
            },
            FlatAttrOptions::Exclude => {
                quote!()
            },
        };
        commands.push(code);
    }

    Ok(quote!(
        impl #generics crate::Flatten for #struct_name #stripped_generics
        #where_clause {
            fn flatten<'a>(&'a mut self, path: String, world: &mut crate::World<'a>) {
                #(#commands)*
            }
        }
    ))
}


struct SimpleDataStruct {
    generics: Generics,
    stripped_generics: Generics,
    where_clause: Option<WhereClause>,
    name: Ident,
}

enum SimpleStructFields {
    Named(Vec<(Ident, Vec<Attribute>)>),
    Unnamed(Vec<Vec<Attribute>>),
    Unit
}

fn extract_fields(derive: &DeriveInput) -> Result<SimpleStructFields> {
    if let Struct(data) = &derive.data {
        match &data.fields {
            Fields::Named(fields) => {
                let fields: Vec<_> = fields.named.iter().filter(|x| x.ident.is_some()).map(|x| {
                    let id = x.ident.clone().unwrap();
                    (id, x.attrs.clone())
                }).collect();
                Ok(SimpleStructFields::Named(fields))
            }
            Fields::Unnamed(fields) => {
                let ranges: Vec<_> = fields.unnamed.iter().map(|x| x.attrs.clone()).collect();
                Ok(SimpleStructFields::Unnamed(ranges))
            }
            Fields::Unit => Ok(SimpleStructFields::Unit),
        }
    } else {
        return Err(Error::msg("Expected a struct"));
    }
}

impl SimpleDataStruct {
    fn new(derive: DeriveInput) -> Result<Self> {
        let stripped_generics = strip_trait_bounds(&derive.generics);
        let where_clause = derive.generics.where_clause.clone();
        Ok(Self {
            generics: derive.generics,
            stripped_generics,
            where_clause,
            name: derive.ident,
        })
    }
}

fn strip_trait_bounds(generics: &Generics) -> Generics {
    let mut generics = generics.clone();
    for g in &mut generics.params {
        match g {
            syn::GenericParam::Type(param) => {
                param.bounds.clear();
            }
            syn::GenericParam::Lifetime(param) => {
                param.bounds.clear();
            }
            syn::GenericParam::Const(_) => {}
        }
    }
    generics
}

#[test]
fn compute_attr_test() {
    let basic_struct = quote!(
        struct S {
            #[flat(skip)]
            a: f32,
            #[skip]
            b: std::rc::Rc(usize),
            #[flat[skip]]
            c: std::collections::Hashmap<i32, String>,
            #[flat(exclude, skip)]
            d: i32,
            #[flat]
            f: i32,
            #[flat(exclude)]
            g: i64,
        }
    );

    let deriveinput = parse2::<DeriveInput>(basic_struct).unwrap();
    let fields = extract_fields(&deriveinput).unwrap();
    if let SimpleStructFields::Named(fields) = fields {
        for f in fields {
            for i in f.1 {
                //println!("{:#?}", i);
                println!("{:?}", compute_attr(&i));
            }
        }
    }
    //println!("{:#?}", deriveinput);
}

#[test]
fn derive_flatten_test() {
    let basic_struct = quote!(
        struct Basic {
            a: Vec<f32>,
            b: HashMap<usize, String>,
            #[flat(skip)]
            c: fn(usize) -> usize,
        }
    );

    let derived = derive_flatten(basic_struct);
    println!("{}", derived.unwrap());
}

#[derive(Debug, PartialEq, Eq)]
enum FlatAttrOptions {
    Skip,
    Exclude,
    Include
}

fn compute_attr(attrs: &Attribute) -> Result<FlatAttrOptions> {
    // first make sure that any attributes are surrounded by flat(...)
    // if let Some(ident) = attrs.path.get_ident() {
    //     if ident.to_string() == "flat".to_string() {
    //         let tokens = attrs.tokens.to_string();
    //         let has_exclude = tokens.contains("exclude");
    //         let has_skip = tokens.contains("skip");
    //         if has_exclude && has_skip {
    //             return Err(Error::msg("flat(...), cannot have both exclude and skip, choose exclude to exclude a field from being included, and skip to stop that field from being flattened."));
    //         } else if has_exclude {
    //             return Ok(FlatAttrOptions::Exclude);
    //         } else if has_skip {
    //             return Ok(FlatAttrOptions::Skip);
    //         } else {
    //             return Err(Error::msg("no option chosen, either choose flat(exclude) to exclude a field from being inserted into world, or choose flat(skip) to prevent that field from being flattened, but it is still inserted into world."));
    //         }
    //     }
    // }
    Ok(FlatAttrOptions::Include)
}

fn compute_attributes(attrs: &[Attribute]) -> Result<FlatAttrOptions> {
    let mut attr_arg = FlatAttrOptions::Include;
    for attr in attrs.iter() {
        let at = compute_attr(attr)?;
        match attr_arg {
            FlatAttrOptions::Include => { 
                    if at != FlatAttrOptions::Include {
                        attr_arg = at;
                    }
                }
            _ => { return Err(Error::msg("only only skip(...) attribute allowed")); }
        } 
    }
    Ok(attr_arg)
}

#[test]
fn test_enum() {
    let a = quote!(
        enum T {
            A,
            B(usize)
        }
    );

    let derived = parse2::<DeriveInput>(a).unwrap();
    println!("{:#?}", derived);
}