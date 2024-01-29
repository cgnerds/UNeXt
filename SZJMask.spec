# -*- mode: python -*-
# vim: ft=python

import sys

sys.setrecursionlimit(5000)  # required on Windows

a = Analysis(
    ['infer_video.py'],
    pathex=[],
    binaries=[('libs/*', '.')],
    datas=[
       ('models/wrist/*', 'models/wrist'),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='SZJMask',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon='libs/icon.icns',
)
app = BUNDLE(
    exe,
    name='USAI.app',
    icon='anylabeling/resources/images/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
