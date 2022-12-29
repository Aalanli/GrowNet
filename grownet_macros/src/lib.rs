use proc_macro::TokenStream;
use grownet_macro_core as macros;

#[proc_macro_derive(Config, attributes(no_op))]
pub fn derive_macro_config(input: TokenStream) -> TokenStream {
    macros::derive_macro_config(input.into()).unwrap().into()
}
