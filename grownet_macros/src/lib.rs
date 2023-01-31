use grownet_macro_core as macros;
use proc_macro::TokenStream;

#[proc_macro_derive(Config, attributes(no_op))]
pub fn derive_macro_config(input: TokenStream) -> TokenStream {
    macros::derive_macro_config(input.into()).unwrap().into()
}

#[proc_macro_derive(UI, attributes(no_op))]
pub fn derive_macro_ui(input: TokenStream) -> TokenStream {
    macros::derive_macro_ui(input.into()).unwrap().into()
}

#[proc_macro]
pub fn derive_ui(input: TokenStream) -> TokenStream {
    macros::derive_macro_ui(input.into()).unwrap().into()
}
