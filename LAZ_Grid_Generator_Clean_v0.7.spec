# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# ezdxf ships data files (fonts, resources) that must travel with the exe
ezdxf_datas, ezdxf_binaries, ezdxf_hiddenimports = collect_all('ezdxf')

a = Analysis(
    ['laz_grid_generator_gui_v0.7.py'],
    pathex=[],
    binaries=ezdxf_binaries,
    datas=ezdxf_datas,
    hiddenimports=ezdxf_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LAZ_Grid_Generator_Clean_v0.7',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.ico'],
)
