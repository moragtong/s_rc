/tmp/opencode/venv/bin/python - <<'EOF'
import numpy as np
from PIL import Image
a = np.fromfile('/tmp/w.bin', dtype=np.float32).reshape(512, 512)
print("flagged:", (np.abs(a) > 0.5).sum())
mask = (np.abs(a) > 0.5).astype(np.uint8) * 255
Image.fromarray(np.flipud(mask)).resize((2048, 2048), Image.Resampling.NEAREST).save('/tmp/w.png')
EOF

# Bake camera orthographic PNG
S_RC_BAKE_VIEW=down S_RC_DIRECT_ONLY=1 S_RC_SCREENSHOT=/tmp/down.png ./target/debug/s_rc
S_RC_BAKE_VIEW=up   S_RC_DIRECT_ONLY=1 S_RC_SCREENSHOT=/tmp/up.png   ./target/debug/s_rc
