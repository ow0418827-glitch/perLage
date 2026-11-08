"""本地商城运行依赖；外部目录只读，所有实验写入当前项目。"""
import os
from src.paths import ProjectPath as Path, PROJECT

CATALOG = Path(os.environ.get('PERLAGE_WEBSHOP_CATALOG',
               PROJECT.parent / 'lage/src/output/nine_group_catalog_full')).resolve()
WEBSHOP_SOURCE = Path(os.environ.get('PERLAGE_WEBSHOP_SOURCE',
                      PROJECT.parent / 'MINJA-main/rap/webshop')).resolve()
