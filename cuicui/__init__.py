"""
CuiCui
======
My inspiration to build this framework came from my two guinea pigs.
The name cuicui comes from the sound they constantly make.
In this project, you can find some Easter eggs related to them,
so have fun trying to discover them! If you want to know more about them,
you can use the variables `cui_01` or `cui_02`.

Original Names:
    - cui_01 is Reginaldo
    - cui_02 is Robson

Now, if you'd like to know more about me:
    - `My WebSite <https://kleversonsantosportfolio.com/>`_
    - `LinkedIn <https://linkedin.com/in/kleverson-santos-a29354182>`_
    - `Medium <https://medium.com/@klevoli>`_
    - `GitHub <https://github.com/Kle-oliver>`_
"""

import importlib.resources
from PIL import Image


with importlib.resources.path('cuicui.utils', 'cui_01.png') as cui_01_path:
    cui_01 = Image.open(cui_01_path)

with importlib.resources.path('cuicui.utils', 'cui_02.png') as cui_02_path:
    cui_02 = Image.open(cui_02_path)
