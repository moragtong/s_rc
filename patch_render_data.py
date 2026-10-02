import sys
code = open("src/cascades2_5d/mod.rs").read()
code = code.replace(
"""struct CascadesRenderData {
    hit_buffer: Buffer,
    sh_probes_buffer: Buffer,
    uniform_buffer: Buffer,
    raymarch_bind_group: BindGroup,
    gather_bind_group: BindGroup,
}""",
"""pub struct CascadesRenderData {
    pub hit_buffer: Buffer,
    pub sh_probes_buffer: Buffer,
    pub uniform_buffer: Buffer,
    pub raymarch_bind_group: BindGroup,
    pub gather_bind_group: BindGroup,
}""")
open("src/cascades2_5d/mod.rs", "w").write(code)
