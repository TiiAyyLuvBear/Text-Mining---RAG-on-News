import json
from pathlib import Path

def load_jsonl(filepath):
    """Load JSONL file and return list of dictionaries"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    qa_file = Path(r"d:\HCMUS\HOCTAP\Semesters\25-26HK3\KhaiThacDuLieuVanBan\Project\Text-Mining---RAG-on-News\src\TEST_OUT\QWEN\data_QA_Convert.jsonl")
    answers_file = Path(r"d:\HCMUS\HOCTAP\Semesters\25-26HK3\KhaiThacDuLieuVanBan\Project\Text-Mining---RAG-on-News\src\TEST_OUT\QWEN\answers_qwen_colab_full.jsonl")
    
    print("=" * 80)
    print("KIEM TRA CAU HOI is_possible=false DUOC TRA LOI")
    print("=" * 80)
    print()
    
    # Load data
    qa_records = load_jsonl(qa_file)
    answer_records = load_jsonl(answers_file)
    
    print(f"Tong so cau hoi trong file QA: {len(qa_records)}")
    print(f"Tong so cau tra loi trong file answers: {len(answer_records)}")
    print()
    
    # Find all is_possible=false questions
    impossible_questions = [q for q in qa_records if q.get('is_possible') is False]
    print(f"So cau hoi co is_possible=false: {len(impossible_questions)}")
    print()
    
    # Build answer lookup by qa_id
    answer_lookup = {a['qa_id']: a for a in answer_records if 'qa_id' in a}
    
    # Check which impossible questions were answered
    matched = []
    for q in impossible_questions:
        qa_id = q.get('id')
        if qa_id and qa_id in answer_lookup:
            answer = answer_lookup[qa_id]
            generated = answer.get('generated_answer', '')
            if generated and generated.strip():
                matched.append({
                    'id': qa_id,
                    'question': q.get('question', ''),
                    'answer': generated.strip()
                })
    
    print("-" * 80)
    print(f"KET QUA: {len(matched)} cau hoi is_possible=false nhung van duoc tra loi")
    print("-" * 80)
    print()
    
    if len(matched) == 0:
        print("Khong co cau hoi nao voi is_possible=false duoc tra loi.")
        print()
        print("Giai thich:")
        print(f"  - File QA co {len(impossible_questions)} cau is_possible=false")
        print(f"  - File answers chi co {len(answer_records)} cau, tat ca deu la is_possible=true")
        print(f"  -> Cac cau is_possible=false da duoc loc bo truoc khi sinh cau tra loi")
    else:
        print("Danh sach cac cau hoi:")
        print()
        for i, item in enumerate(matched, 1):
            print(f"{i}. ID: {item['id']}")
            print(f"   Cau hoi: {item['question']}")
            print(f"   Tra loi: {item['answer'][:200]}{'...' if len(item['answer']) > 200 else ''}")
            print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()
