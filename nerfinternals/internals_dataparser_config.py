from __future__ import annotations

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfinternals.llff_dataparser import LLFFDataParserConfig

llff_data = DataParserSpecification(config=LLFFDataParserConfig())