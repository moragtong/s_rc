use std::fs;

fn main() {
    let mut code = fs::read_to_string("src/heightmap.rs").unwrap();
    code = code.replace(
        "let bytes: [u8; 16] = [0u8; 16]; // min",
        "//..",
    );
}
