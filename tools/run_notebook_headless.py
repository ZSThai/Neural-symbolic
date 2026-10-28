import os
import sys
import json
import time
from types import ModuleType

os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get('TEMP', '.'), 'mpl_cfg')

# Mock IPython for headless terminal execution
def dummy_display(*args, **kwargs):
    for a in args:
        if hasattr(a, 'to_string'):
            print(a.to_string())
        elif isinstance(a, str):
            print(a)

def dummy_markdown(s):
    return s

ip_mod = ModuleType('IPython')
ip_mod.get_ipython = lambda: None
ip_mod.version_info = (8, 24, 0)
ip_disp = ModuleType('IPython.display')
ip_disp.display = dummy_display
ip_disp.Markdown = dummy_markdown
sys.modules['IPython'] = ip_mod
sys.modules['IPython.display'] = ip_disp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: None

from pathlib import Path
step1_dir = Path(r"c:\Users\letha\Downloads\New Paper with Umberto\claude_research\step1")
os.chdir(step1_dir)
sys.path.insert(0, str(step1_dir))

nb = json.loads((step1_dir / 'Step1_CBIS_DDSM_Colab.ipynb').read_text(encoding='utf-8'))
code_cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

g = {'__name__': '__main__', 'display': dummy_display, 'Markdown': dummy_markdown}
print(f"Bắt đầu chạy 21 cells của Bước 1...")
t0 = time.time()

for idx, code in enumerate(code_cells, 1):
    c_t0 = time.time()
    clean_lines = []
    for line in code.splitlines():
        if line.strip().startswith('%') or line.strip().startswith('!'):
            continue
        clean_lines.append(line)
    clean_code = '\n'.join(clean_lines)
    try:
        exec(clean_code, g)
        print(f"  [Cell {idx}/21] OK ({round(time.time() - c_t0, 2)}s)")
    except Exception as e:
        print(f"  [Cell {idx}/21] ERROR: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\nTổng thời gian chạy: {round(time.time() - t0, 2)}s")
