import sys
code = open("src/main.rs").read()
code = code.replace("Cascades2_5dPlugin,", "Cascades2_5dPlugin,\n            shade::ShadePlugin,")
open("src/main.rs", "w").write(code)
