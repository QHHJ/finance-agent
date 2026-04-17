import os
from pathlib import Path
from app.services import parser, extractor

pdf=Path(r"E:\xwechat_files\wxid_jdvxxma8fib922_1e53\msg\file\2026-04\dzfp_26222000000256887286_吉林大学_20260331140832(1).pdf")
text=parser.parse_pdf_text(pdf)

os.environ['USE_OLLAMA_VL']='true'
os.environ['LONG_MODE_USE_TEXT_LLM']='false'
res=extractor.extract_invoice_fields(text,pdf_path=str(pdf))
print('source=',res.get('extraction_source'))
print('mode=',res.get('processing_mode'))
print('stats=',res.get('long_mode_stats'))
print('warnings=',res.get('warnings'))
items=res.get('line_items') or []
print('rows=',len(items))
for i,row in enumerate(items[:20],1):
    print(i,row.get('item_name'), '| spec=',row.get('spec'),'| qty=',row.get('quantity'),'| unit=',row.get('unit'),'| total=',row.get('line_total_with_tax'))
print('amount=',res.get('amount'),'tax=',res.get('tax_amount'))
