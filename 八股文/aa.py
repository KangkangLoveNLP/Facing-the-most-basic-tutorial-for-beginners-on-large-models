import fitz  # PyMuPDF

def split_pdf(input_pdf_path, output_pdf1_path, output_pdf2_path, split_page_number):
    # 打开PDF文件
    pdf_document = fitz.open(input_pdf_path)
    
    # 获取总页数
    total_pages = pdf_document.page_count
    
    # 检查分割页号是否有效
    if split_page_number < 1 or split_page_number > total_pages:
        raise ValueError("Split page number is out of range.")
    
    # 创建第一个PDF文档（包含第1页到分割页）
    pdf_document1 = fitz.open()
    for page_num in range(split_page_number):
        pdf_document1.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
    
    # 创建第二个PDF文档（包含分割页之后的页面）
    pdf_document2 = fitz.open()
    for page_num in range(split_page_number, total_pages):
        pdf_document2.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
    
    # 保存两个PDF文档
    pdf_document1.save(output_pdf1_path)
    pdf_document2.save(output_pdf2_path)
    
    # 关闭PDF文档
    pdf_document.close()
    pdf_document1.close()
    pdf_document2.close()

# 示例用法
input_pdf_path = r"D:\User\李盛康\Downloads\笔迹心理学 (郑日昌) (Z-Library).pdf"  # 输入PDF文件路径
output_pdf1_path = "output_part1.pdf"  # 第一部分输出PDF文件路径
output_pdf2_path = "output_part2.pdf"  # 第二部分输出PDF文件路径
split_page_number = 50  # 从第1245页开始分割

split_pdf(input_pdf_path, output_pdf1_path, output_pdf2_path, split_page_number)