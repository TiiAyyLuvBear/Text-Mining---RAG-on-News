# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT

## Cách sử dụng tài liệu

Đây là bản thuyết minh đầy đủ theo từng slide. Nếu đọc toàn bộ, thời lượng sẽ dài hơn 20 phút. Khi thuyết trình 20 phút, ưu tiên các đoạn có nhãn “Lời trình bày chính”; các đoạn “Giải thích thêm” dùng khi còn thời gian hoặc khi thầy cô đặt câu hỏi.

Các thuật ngữ tiếng Anh như retrieval, chunking, embedding, reranking và generation được giữ lại vì đây là tên kỹ thuật xuất hiện trực tiếp trên slide. Khi nói, nên giải thích ngắn bằng tiếng Việt ở lần xuất hiện đầu tiên.

---

## Mở đầu

### Lời trình bày chính

Kính chào thầy cô và các bạn.

Hôm nay em xin trình bày báo cáo giữa kỳ của đề tài Text Mining – Retrieval-Augmented Generation on Vietnamese News. Mục tiêu của đề tài là xây dựng một hệ thống hỏi đáp trên dữ liệu tin tức tiếng Việt bằng kiến trúc RAG.

Điểm quan trọng của hệ thống là mô hình ngôn ngữ không trả lời chỉ dựa trên kiến thức đã học trước đó. Trước khi sinh câu trả lời, hệ thống phải truy xuất các đoạn tin tức có liên quan trong corpus và dùng chính các đoạn này làm bằng chứng. Nhờ vậy, câu trả lời có căn cứ rõ hơn và giảm nguy cơ hallucination.

Trong phần trình bày, em sẽ đi từ dữ liệu, pipeline tạo QA và kiến trúc tổng thể; sau đó trình bày các thí nghiệm về chunking, embedding, reranking và generation; cuối cùng là kết quả, hạn chế và kế hoạch hoàn thiện hệ thống.

### Giải thích thêm

RAG là viết tắt của Retrieval-Augmented Generation. “Retrieval” là bước tìm lại thông tin liên quan, còn “Generation” là bước dùng mô hình ngôn ngữ để tạo câu trả lời. Nếu retrieval sai thì dù generator mạnh, câu trả lời vẫn có thể sai vì context không chứa bằng chứng. Vì vậy đề tài này không chỉ đánh giá chatbot ở đầu ra cuối cùng mà còn đánh giá riêng từng tầng của pipeline.

---

# 1. Problem, Dataset & Architecture

## Slide: Problem Statement

### Lời trình bày chính

Bài toán của nhóm là hỏi–đáp trên tin tức tiếng Việt. Đầu vào của hệ thống là một câu hỏi của người dùng. Hệ thống cần tìm đúng thông tin trong tập bài báo, sau đó sử dụng thông tin đó để tạo ra câu trả lời ngắn gọn, tự nhiên và có căn cứ.

Điểm quan trọng của RAG là chất lượng câu trả lời phụ thuộc trực tiếp vào chất lượng truy xuất. Nếu hệ thống tìm sai bài báo hoặc sai đoạn văn thì dù mô hình sinh mạnh đến đâu, câu trả lời vẫn có thể không chính xác.

Bài toán này có bốn thách thức chính. Thứ nhất, corpus có nhiều bài báo dài và thuộc nhiều chủ đề khác nhau. Thứ hai, câu hỏi thường chỉ liên quan đến một phần nhỏ trong bài, nên tìm đúng bài vẫn chưa đủ mà phải tìm đúng chunk chứa bằng chứng. Thứ ba, các embedding model biểu diễn tiếng Việt khác nhau nên kết quả xếp hạng cũng khác nhau. Cuối cùng, câu trả lời sinh tự nhiên không thể chỉ đánh giá bằng exact match, vì hai câu có thể diễn đạt khác nhau nhưng vẫn cùng ý nghĩa.

Vì vậy, nhóm đánh giá riêng cả retrieval lẫn generation, trong đó đặc biệt chú ý đến tính chính xác và faithfulness của câu trả lời đối với context.

### Câu chuyển

Để xây dựng và đánh giá hệ thống này, nhóm sử dụng hai thành phần dữ liệu: corpus bài báo và tập câu hỏi–trả lời.

### Giải thích thêm

Trong miền tin tức, các chi tiết như tên người, tổ chức, ngày tháng, địa điểm và số liệu rất quan trọng. Một câu trả lời nghe hợp lý nhưng sai một con số hoặc nhầm sự kiện vẫn là câu trả lời sai. Do đó, faithfulness với context là tiêu chí đặc biệt quan trọng.

---

## Slide: Dataset Overview

### Lời trình bày chính

Dữ liệu chính của đề tài là VietOnlineNews. Corpus được đưa vào pipeline chunking và indexing gồm 10.073 bài báo, tương ứng với 10.073 article ID duy nhất.

Bên cạnh corpus, nhóm xây dựng một tập đánh giá gồm 152 cặp câu hỏi–trả lời. Hai tập này có vai trò khác nhau: corpus cung cấp không gian để hệ thống truy xuất, còn QA set cung cấp câu hỏi và ground truth để kiểm tra hệ thống có tìm đúng nguồn và trả lời đúng hay không.

Trong quá trình thực nghiệm, nhóm tạo ra các artifact gồm cleaned corpus, các phiên bản chunking, vector embedding, kết quả retrieval, kết quả reranking, generated answers và các bảng metric cho cả retrieval lẫn generation.

Luồng dữ liệu trên slide mô tả ngắn gọn quá trình từ bài báo thô, qua làm sạch và chia chunk, sau đó tạo vector index và thực hiện đánh giá. Tuy nhiên, cần hiểu rằng QA benchmark là một nhánh đánh giá độc lập, không phải dữ liệu được sinh ra từ vector index.

### Câu chuyển

Tiếp theo, em trình bày cụ thể cách nhóm xây dựng tập QA dùng làm benchmark.

### Giải thích thêm

Data flow trên slide có thể được hiểu theo hai nhánh:

- Nhánh corpus: Raw News → Cleaned Articles → Chunks → Dense Embeddings → Retrieval.
- Nhánh benchmark: Cleaned Articles → QA Generation → QA Validation → Evaluation Queries.

Hai nhánh gặp nhau ở bước evaluation. Vì vậy QA không phải là sản phẩm được sinh ra từ vector index; QA là bộ truy vấn dùng để đánh giá index.

---

## Slide: Pipeline QA

### Lời trình bày chính

Pipeline tạo QA gồm hai nhánh: single-article và cross-article.

Ở nhánh single-article, tiêu đề và nội dung của từng bài báo được đưa vào Claude để sinh câu hỏi. Các câu hỏi được chia thành bảy loại, gồm factoid, event summary, cause–effect, entity role, comparison, claim verification và unanswerable. Nhờ đó, benchmark không chỉ kiểm tra khả năng tìm một thông tin đơn giản mà còn kiểm tra khả năng tóm tắt, suy luận nguyên nhân–kết quả và xác minh nhận định.

Ở nhánh cross-article, nhóm lấy các bài cùng category, biểu diễn chúng bằng TF–IDF và sử dụng clustering để tìm các bài có nội dung liên quan. Từ mỗi nhóm bài, Claude sinh các câu hỏi multi-document comparison và timeline. Đây là những câu khó hơn vì hệ thống phải tổng hợp bằng chứng từ nhiều bài báo.

Sau khi hợp nhất hai nhánh, chương trình chuẩn hóa `qa_type` và `is_possible`, đồng thời gắn `article_id` hoặc danh sách `source_article_ids` làm ground truth. Kết quả được xuất thành `QA_output.jsonl` và được kiểm tra tự động về schema, ID, trường bắt buộc, answerability và liên kết với corpus.

Hiện tại tập QA đã qua validation tự động. Bước human review vẫn cần được hoàn thiện trước khi chốt benchmark cuối cùng. Đặc biệt, các câu unanswerable giúp kiểm tra hệ thống có biết từ chối khi không đủ bằng chứng hay không, thay vì cố tạo ra một câu trả lời.

### Câu chuyển

Sau khi có corpus và QA benchmark, nhóm thực hiện EDA để hiểu rõ chất lượng và phân bố dữ liệu trước khi chạy retrieval.

### Giải thích thêm

Pipeline QA độc lập với pipeline chunking. Script tạo QA đọc tiêu đề và nội dung bài báo; nó không lấy chunk làm đầu vào và cũng không có bước extract facts riêng. Chunking được thực hiện sau đó để tạo đơn vị retrieval.

Script tạo QA trong repo đang cấu hình model Claude Opus 4.6. Đây là model dùng để tạo benchmark. Model dùng ở bước sinh câu trả lời cuối pipeline là Claude Opus 4.8. Hai bước này cần được phân biệt khi thuyết trình.

Việc có câu unanswerable giúp kiểm tra hệ thống có biết từ chối khi context không đủ hay không. Đây là tình huống quan trọng đối với RAG vì hệ thống tốt không chỉ cần trả lời đúng khi có bằng chứng, mà còn cần tránh bịa khi không có bằng chứng.

---

## Slide: EDA corpus tin tức

### Lời trình bày chính

Đối với corpus, tập train dùng cho indexing có 10.073 bài báo. Notebook EDA đồng thời kiểm tra thêm validation gồm 2.500 bài và test gồm 2.000 bài.

Kết quả cho thấy dữ liệu có 13 category. Các trường quan trọng như ID, tiêu đề, mô tả, nội dung và category không bị thiếu, đồng thời không có article ID trùng. Tuy nhiên, phân bố giữa các chủ đề chưa đồng đều. Ví dụ, Giải trí chiếm tỷ lệ lớn nhất, trong khi Pháp luật có số lượng bài ít hơn đáng kể.

Sự mất cân bằng này có thể làm cho metric trung bình che khuất hiệu năng ở từng nhóm dữ liệu. Hệ thống có thể truy xuất tốt ở các chủ đề phổ biến nhưng kém ổn định ở những category ít dữ liệu. Vì vậy, ngoài kết quả tổng thể, nhóm cần theo dõi retrieval theo từng category.

Độ dài bài báo cũng biến thiên khá lớn. Một bài ngắn có thể chỉ tạo một chunk, trong khi bài dài tạo ra nhiều chunk hơn. Điều này ảnh hưởng trực tiếp đến số vector, không gian tìm kiếm và khả năng đưa đúng evidence vào top-k.

### Câu chuyển

Với corpus đã được khảo sát, phần tiếp theo là đặc điểm của 152 câu hỏi dùng để đánh giá hệ thống.

### Giải thích thêm

EDA còn phát hiện một số record có dấu hiệu mojibake hoặc lỗi encoding. Vì vậy cách diễn đạt chính xác là nhóm đã chuẩn hóa và kiểm tra encoding, không nên nói rằng mọi lỗi encoding đã được loại bỏ hoàn toàn.

Độ dài bài báo ảnh hưởng trực tiếp đến chunking. Bài ngắn có thể chỉ tạo một chunk, còn bài dài tạo nhiều chunk. Nếu một category có các bài đặc biệt dài, số vector của category đó cũng lớn hơn và có thể ảnh hưởng retrieval.

---

## Slide: EDA tập câu hỏi–trả lời

### Lời trình bày chính

Tập đánh giá gồm 152 QA với 152 ID duy nhất và không có câu hỏi trùng hoàn toàn.

Trong số này, 114 câu có thể trả lời dựa trên dữ liệu, còn 38 câu được gắn `is_possible` bằng false. Nhóm 38 câu này bao gồm các trường hợp không có đủ bằng chứng, vì vậy hệ thống được kỳ vọng từ chối hoặc thông báo rằng context không đủ.

Độ dài câu hỏi trung bình khoảng 32 từ, trong khi câu trả lời trung bình khoảng 48 từ nếu tính trên toàn bộ tập, bao gồm cả những answer rỗng. Như biểu đồ cho thấy, câu trả lời thường dài hơn và có thể được diễn đạt theo nhiều cách khác nhau.

Do đó, exact match hoặc BLEU không đủ để phản ánh toàn bộ chất lượng generation. Nhóm cần kết hợp metric bề mặt với metric ngữ nghĩa và LLM Judge. Đồng thời, 114 câu answerable nên được dùng để đánh giá khả năng tìm evidence, còn 38 câu unanswerable nên được phân tích riêng bằng refusal accuracy.

### Câu chuyển

Ngoài độ dài, loại câu hỏi và số lượng nguồn cần tổng hợp cũng quyết định độ khó của benchmark.

### Giải thích thêm

Retrieval evaluation hợp lý nhất nên tập trung vào 114 câu answerable khi đo khả năng tìm evidence. Với 38 câu unanswerable, metric quan trọng hơn là refusal accuracy, tức hệ thống có từ chối đúng hay không. Nếu trộn hai nhóm vào cùng một metric mà không giải thích, kết quả có thể khó diễn giải.

---

## Slide: EDA loại câu hỏi và nguồn dữ liệu

### Lời trình bày chính

QA set bao gồm cả câu hỏi đơn bài, đa bài và câu hỏi thiếu bằng chứng.

Nhóm lớn nhất là multi-document comparison với 42 câu. Ngoài ra có 21 câu cause–effect, 20 factoid, 20 event summary, 14 entity role, 10 timeline và một số câu comparison, claim verification và unanswerable.

Phân bố này cho thấy benchmark không chỉ đo khả năng trích xuất một thông tin đơn giản. Nhiều câu yêu cầu hệ thống tổng hợp, so sánh hoặc sắp xếp sự kiện từ nhiều bài báo. Toàn bộ 13 category của corpus đều xuất hiện trong nguồn QA, tuy nhiên mức độ đại diện giữa các category vẫn chưa hoàn toàn cân bằng.

Đặc biệt, với câu hỏi multi-document, việc tìm đúng một bài chưa có nghĩa là đã đủ bằng chứng. Top-k phải bao phủ được nhiều source article cần thiết. Vì vậy, kết quả cuối cùng nên được phân tích riêng theo QA type, single-doc và multi-doc, cũng như theo answerability.

### Câu chốt phần

Tóm lại, dữ liệu được thiết kế để kiểm tra cả ba khả năng: tìm đúng nguồn, tổng hợp đúng bằng chứng và biết từ chối khi bằng chứng không đầy đủ. Đây là cơ sở để nhóm xây dựng và đánh giá kiến trúc RAG ở phần tiếp theo.

### Giải thích thêm

Cross-article QA khó hơn vì top-k cần bao phủ nhiều article ID. Một hệ thống có thể tìm đúng một bài nhưng chưa đủ bằng chứng để trả lời toàn bộ câu hỏi. Vì vậy Recall@k và cách định nghĩa relevance cho multi-doc cần được ghi rõ trong báo cáo.

---

## Slide: Data Processing Pipeline

### Lời trình bày chính

Quy trình xử lý dữ liệu bắt đầu từ VietOnlineNews, sau đó chuẩn hóa file và các trường id, title, description, content và category. Metadata được giữ lại để sau này mỗi chunk có thể truy ngược về bài báo nguồn.

Từ corpus đã chuẩn hóa, nhóm thực hiện EDA, tạo QA, validation QA và tạo bốn phiên bản chunking. Các chunk tiếp tục được embedding, retrieval và reranking.

Script tạo QA hiện cấu hình Claude Opus 4.6. QA_output.jsonl hiện được dùng trong thí nghiệm. Trước báo cáo cuối kỳ, nhóm cần human review, sửa hoặc loại QA lỗi và đóng băng một benchmark version cố định.

### Giải thích thêm

Metadata quan trọng vì câu trả lời RAG cần citation. Nếu chỉ lưu text của chunk mà không giữ article_id, title hoặc URL, hệ thống khó chứng minh nguồn. Trong thí nghiệm hiện tại, article_id là trường chủ yếu dùng để tính ground truth retrieval.

---

## Slide: Overall RAG Architecture

### Lời trình bày chính

Hệ thống gồm hai pha chính.

Ở pha offline indexing, các bài báo đã làm sạch được chia thành chunk. Mỗi chunk được embedding thành một vector ngữ nghĩa, sau đó vector và metadata như article ID, title và URL được lưu lại phục vụ truy xuất.

Ở pha online, câu hỏi được embedding bằng cùng model. Hệ thống tính độ tương đồng giữa query vector và các chunk vector để lấy top-k candidate. Các candidate tiếp tục được chấm lại bằng BGE hoặc Jina reranker. Cuối cùng, top-5 context sau reranking được đưa vào Claude để sinh câu trả lời.

Vì code hiện tại chỉ sử dụng dense vector retrieval rồi semantic reranking, cách gọi chính xác là two-stage dense semantic retrieval. Đây chưa phải hybrid retrieval vì pipeline chưa kết hợp BM25 hoặc sparse retrieval.

### Giải thích thêm

BGE-M3 có thể hỗ trợ nhiều cơ chế biểu diễn trong các cách triển khai khác, nhưng trong code hiện tại nó được gọi qua SentenceTransformer.encode và tạo một dense vector cho mỗi text. Đồ án chưa sử dụng sparse output hoặc multi-vector retrieval của BGE-M3.

Embedding model và reranker là hai loại model khác nhau. BAAI/bge-m3 trên slide embedding dùng để tạo vector. BAAI/bge-reranker-v2-m3 dùng để chấm cặp query–document. Không nên gọi hai model này là cùng một thành phần.

### Lưu ý khi trình bày

- 10.073 là số bài train dùng làm corpus chính cho chunking và indexing. Notebook EDA kiểm tra cả ba split, tổng cộng 14.573 bài.
- Các thống kê 32 và 48 trong notebook là số từ theo phép tách khoảng trắng; nên nói “từ”, không nên khẳng định là tokenizer token.
- Chỉ nói QA “đã qua validation tự động và cần human review”, không nói toàn bộ 152 câu đã được con người xác nhận.
- Không gọi pipeline hiện tại là hybrid retrieval.

---

## Slide: Experimental Design

### Lời trình bày chính

Trong experimental design, các thành phần cố định gồm corpus, QA evaluation set, cách tính metric, generator và format output.

Các biến thay đổi gồm bốn embedding model: Alibaba-NLP/gte-multilingual-base, BAAI/bge-m3, Multilingual E5-large và Qwen3 Embedding 0.6B. Nhóm so sánh chủ yếu trên Token và Structured chunking, sau đó thử BGE và Jina reranker.

Mục tiêu là xác định embedding nào truy xuất evidence tốt nhất, chunking nào cân bằng chất lượng và chi phí, reranker nào đưa evidence đúng lên đầu, và các thay đổi retrieval ảnh hưởng thế nào đến câu trả lời cuối.

### Giải thích thêm

Để so sánh công bằng, tất cả model cần chạy trên cùng QA set, cùng top-k, cùng ground truth và cùng định nghĩa metric. Đây cũng là lý do nhóm liệt kê “chuẩn hóa benchmark” trong phần kế hoạch.

---

# 2. Experiments & Results

## Slide: Chunking Strategies

### Lời trình bày chính

Nhóm thử bốn chiến lược chunking trên 10.073 bài với cấu hình mục tiêu khoảng 450 token, overlap 80 và có chèn title, description vào chunk.

Token chunking tạo 21.454 chunk, trung bình 2,13 chunk mỗi bài và khoảng 364,65 token mỗi chunk. Thời gian tạo khoảng 3,15 giây.

Structured chunking tạo 22.353 chunk, trung bình 2,22 chunk mỗi bài và khoảng 364,70 token mỗi chunk. Thời gian khoảng 5,42 giây.

LlamaIndex tạo 43.038 chunk, trung bình 4,27 chunk mỗi bài và khoảng 173 token mỗi chunk. LangChain Recursive tạo 85.426 chunk, trung bình 8,48 chunk mỗi bài và chỉ khoảng 84 token mỗi chunk.

Token và Structured vì vậy cân bằng hơn giữa độ đầy đủ context, số lượng vector và chi phí retrieval. LangChain Recursive tạo các đoạn chi tiết nhưng làm số vector tăng gần bốn lần so với Token, tăng kích thước index và có nguy cơ thiếu context.

Do đó, các thí nghiệm reranking chính tập trung vào Token và Structured.

### Giải thích thêm

Overlap giúp giữ thông tin ở ranh giới chunk. Nếu không có overlap, một câu hỏi có thể cần hai câu nằm ở hai chunk khác nhau. Tuy nhiên overlap quá lớn làm dữ liệu lặp và tăng index.

Token chunking chia theo giới hạn token tương đối ổn định. Structured chunking cố giữ cấu trúc đoạn hoặc ranh giới nội dung. Recursive chunking ưu tiên các separator nhỏ hơn nên tạo nhiều đoạn ngắn.

---

## Slide: Embedding Token Comparison

### Lời trình bày chính

Trên Token chunking, nhóm so sánh bốn embedding model theo nDCG@10, Recall@10, Recall@5, MRR@10, Hit@1, Hit@5, latency và index size.

Alibaba GTE đạt nDCG@10 bằng 0,545, Hit@1 bằng 0,461 và có latency thấp nhất khoảng 4,4 mili giây. Model này nhanh và index nhỏ nhưng chất lượng thấp hơn ba model còn lại.

BGE-M3 đạt nDCG@10 bằng 0,772, Recall@10 bằng 0,853 và Hit@5 bằng 0,901. Đây là kết quả mạnh với index khoảng 83,8 MB.

E5-large đạt nDCG@10 cao nhất là 0,788, MRR@10 cao nhất là 0,839, Hit@1 bằng 0,763 và Hit@5 bằng 0,934. Điều này cho thấy E5 thường đưa article đúng lên vị trí rất cao.

Qwen3-0.6B đạt Recall@5 cao nhất là 0,848 nhưng latency khoảng 78 mili giây, cao nhất trong bảng.

Nếu ưu tiên chất lượng retrieval, E5-large nổi bật nhất. Nếu cân bằng chất lượng, latency và index size, BGE-M3 là lựa chọn đáng chú ý.

### Giải thích thêm

Hit@1 cho biết tỷ lệ câu hỏi có kết quả đúng ngay vị trí đầu. MRR quan tâm vị trí của kết quả đúng đầu tiên. Recall@k quan tâm độ bao phủ trong top-k. Trong RAG, Hit@1 và MRR quan trọng vì generator thường chịu ảnh hưởng lớn bởi các context đầu tiên.

Theo bảng hiện tại, E5-large có index lớn hơn. Trước khi giải thích nguyên nhân, nhóm cần kiểm tra lại embedding dimension, kiểu dữ liệu và cách lưu vector của từng lần chạy; kết luận an toàn ở đây là E5 cho chất lượng cao nhưng artifact index được báo cáo tốn bộ nhớ hơn.

---

## Slide: Embedding Structured Comparison

### Lời trình bày chính

Với Structured chunking, xu hướng gần giống Token.

E5-large tiếp tục có nDCG@10 cao nhất là 0,778, MRR@10 là 0,826 và Hit@5 là 0,934. Qwen3 có Recall@5 cao là 0,833 nhưng latency khoảng 77,7 mili giây. BGE-M3 vẫn cân bằng về chất lượng và index size, còn Alibaba GTE nhanh nhưng retrieval thấp hơn.

Khi so Token với Structured, chênh lệch của E5 không quá lớn. Token có nDCG và MRR nhỉnh hơn, đồng thời tạo ít chunk và index nhỏ hơn. Vì vậy Token là lựa chọn hiệu quả hơn cho các bước tiếp theo.

### Giải thích thêm

Không nên kết luận Structured luôn kém. Một số pipeline Structured có Hit@5 tốt, nghĩa là vẫn bao phủ evidence trong top-5. Tuy nhiên khi xét đồng thời rank đầu, số chunk và chi phí, Token ổn định hơn trong thí nghiệm hiện tại.

---

## Slide: Reranking Stage

### Lời trình bày chính

Dense retrieval so sánh vector toàn cục nên có thể lấy được đoạn gần về chủ đề nhưng không chứa đáp án cụ thể. Reranker được thêm vào để chấm trực tiếp mức phù hợp của từng cặp question–chunk.

Nhóm sử dụng hai reranker:

- BAAI/bge-reranker-v2-m3;
- jinaai/jina-reranker-v2-base-multilingual.

Kết hợp hai reranker với hai loại chunking tạo bốn pipeline: BGE_TOKEN, BGE_STRUCTURE, JINA_TOKEN và JINA_STRUCTURE. Candidate từ dense retrieval được chấm lại, sắp xếp theo rerank score và giữ top-5 cho generator.

Mục tiêu của reranking là đưa evidence đúng lên đầu, giảm context nhiễu và gián tiếp tăng correctness, faithfulness của câu trả lời.

### Giải thích thêm

Reranker không tạo lại embedding index. Nó chỉ xử lý một tập candidate nhỏ hơn do dense retrieval cung cấp. Vì vậy kiến trúc hai tầng tiết kiệm hơn việc cho cross-encoder so sánh câu hỏi với toàn bộ hàng chục nghìn chunk.

---

## Slide: Retrieval Metrics – Alibaba GTE

### Lời trình bày chính

Với Alibaba GTE, các pipeline Token tốt hơn Structured.

BGE_TOKEN đạt Hit@1 khoảng 0,566, MRR@5 khoảng 0,588 và nDCG@5 khoảng 0,594, cao nhất hoặc gần cao nhất trong bảng. JINA_TOKEN chỉ nhỉnh nhẹ ở Recall@5.

Kết quả cho thấy Token chunking phù hợp hơn với Alibaba GTE trong thí nghiệm này. Tuy nhiên chất lượng chung vẫn thấp hơn E5-large và BGE-M3, nên Alibaba phù hợp hơn khi ưu tiên tốc độ và index nhỏ.

### Giải thích thêm

Không nên nói Jina hay BGE luôn tốt hơn chỉ từ bảng Alibaba. Sự tương tác giữa embedding, chunking và reranker có thể khác nhau. Reranker nhận candidate do embedding cung cấp; nếu evidence không có trong candidate ban đầu thì reranker không thể phục hồi.

---

## Slide: Retrieval Metrics – E5-Large

### Lời trình bày chính

Với E5-large, bốn pipeline đều đạt kết quả cao.

JINA_TOKEN nổi bật nhất ở các metric chú trọng thứ hạng đầu: Hit@1 bằng 0,842, MRR@5 bằng 0,882 và nDCG@5 bằng 0,809.

BGE_STRUCTURE đạt Hit@5 và Recall@5 cao nhất trong bảng, nghĩa là có độ bao phủ evidence tốt trong năm vị trí đầu. Tuy nhiên JINA_TOKEN đưa evidence đúng lên vị trí đầu thường xuyên hơn.

Trong RAG, context đầu tiên có ảnh hưởng lớn đến generator. Vì vậy E5-large kết hợp Token chunking và Jina reranker là cấu hình retrieval nổi bật nhất hiện tại.

### Giải thích thêm

“Nổi bật nhất” ở đây là theo retrieval ranking, không có nghĩa chắc chắn là tốt nhất end-to-end. Generation còn phụ thuộc prompt, context ordering, độ dài answer và khả năng sử dụng evidence của Claude.

---

## Slide: Retrieval Metrics – Qwen3-0.6B

### Lời trình bày chính

Với Qwen3, BGE_TOKEN đạt Hit@1 bằng 0,822, MRR@5 bằng 0,869 và nDCG@5 bằng 0,789, tốt nhất trên các metric xếp hạng chính.

JINA_TOKEN đạt Hit@5 cao nhất là 0,934. Hai pipeline Token đều tốt hơn Structured trong phần lớn metric. Tuy nhiên Qwen có latency embedding và query cao hơn, nên lợi ích chất lượng cần được cân bằng với tốc độ.

### Giải thích thêm

Kết quả này cũng cho thấy reranker tốt nhất có thể phụ thuộc embedding. Với E5, Jina Token nổi bật; với Qwen, BGE Token tốt hơn về rank đầu.

---

## Slide: Retrieval Metrics – BGE-M3

### Lời trình bày chính

Với BGE-M3, JINA_TOKEN đạt kết quả tốt nhất hoặc gần tốt nhất ở nhiều metric: Hit@1 bằng 0,809, Hit@5 bằng 0,908, Recall@5 bằng 0,839, MRR@5 bằng 0,854 và nDCG@5 bằng 0,818.

BGE_TOKEN cũng có kết quả cạnh tranh, đặc biệt Recall@5 khoảng 0,838 và nDCG@5 khoảng 0,816.

BGE-M3 vì vậy là embedding mạnh và cân bằng. So với E5-large, nó có chất lượng retrieval gần cạnh tranh nhưng index nhỏ hơn.

### Giải thích thêm

Nếu ưu tiên metric nDCG@5 trong bảng sau reranking, BGE-M3 JINA_TOKEN đạt 0,818, cao hơn con số 0,809 của E5 JINA_TOKEN. Tuy nhiên các bảng cần được bảo đảm chạy trên cùng tập truy vấn và cùng cấu hình trước khi tuyên bố model tổng thể tốt nhất. Vì vậy phần kết luận hiện tại vẫn nên dùng từ “nổi bật” và “đề xuất tạm thời”.

---

## Slide: Tổng hợp Retrieval

### Lời trình bày chính

Tổng hợp các thí nghiệm retrieval cho thấy ba điểm.

Thứ nhất, Token chunking thường ổn định và hiệu quả về số chunk, index size lẫn rank đầu.

Thứ hai, E5-large có chất lượng dense retrieval rất mạnh, đặc biệt Hit@1 và MRR.

Thứ ba, BGE-M3 có trade-off tốt giữa chất lượng và tài nguyên. Không có một reranker luôn thắng với mọi embedding: Jina nổi bật với E5 và BGE-M3, còn BGE reranker nổi bật với Qwen.

---

## Slide: Generation Setup

### Lời trình bày chính

Sau retrieval và reranking, top-5 context được đưa vào Claude Opus 4.8 thông qua Anthropic-compatible Messages API.

Input gồm question và các context đã retrieval, rerank. Prompt yêu cầu mô hình chỉ sử dụng context được cung cấp, không suy đoán hoặc bổ sung kiến thức ngoài context, trả lời bằng tiếng Việt và từ chối nếu không đủ bằng chứng.

Output là generated_answer. Mỗi pipeline tạo 152 prediction tương ứng với toàn bộ QA set.

### Giải thích thêm

Prompt grounded generation giúp giảm hallucination nhưng không bảo đảm hoàn toàn. Nếu context sai hoặc chứa nhiều đoạn nhiễu, Claude vẫn có thể chọn nhầm thông tin. Vì vậy cần đánh giá generation cùng với retrieval.

Với unanswerable QA, câu trả lời đúng về hành vi có thể là từ chối. Do đó ngoài BLEU và BERTScore, báo cáo cuối kỳ nên có refusal accuracy.

---

## Slide: Automatic Evaluation Metrics

### Lời trình bày chính

Nhóm dùng bốn nhóm metric.

BLEU đo overlap n-gram giữa generated answer và gold answer. BLEU cao khi các cụm từ trùng nhau nhiều, nhưng có thể thấp dù câu trả lời đúng nghĩa.

ROUGE-L dựa trên longest common subsequence, phản ánh mức độ giữ lại chuỗi nội dung chính.

BERTScore dùng embedding để đo tương đồng ngữ nghĩa, phù hợp hơn khi hai câu dùng từ khác nhau nhưng cùng ý.

LLM Judge chấm câu trả lời từ 1 đến 5 theo correctness, faithfulness, completeness, relevance và fluency. GPTScore là điểm tổng hợp các tiêu chí này.

Không metric nào đủ một mình. Kết luận cần kết hợp metric bề mặt, metric ngữ nghĩa và judge score.

### Giải thích thêm

Correctness so với gold answer; faithfulness kiểm tra câu trả lời có được context hỗ trợ hay không; completeness đo mức đầy đủ; relevance đo đúng trọng tâm; fluency đo độ tự nhiên.

Một câu có thể faithfulness cao nhưng correctness thấp nếu context retrieval bị sai. Khi đó generator trung thành với context nhưng context không chứa gold evidence. Đây là cách phân biệt retrieval error và generation error.

---

## Slide: Generation Results – Alibaba GTE

### Lời trình bày chính

Với Alibaba GTE, JINA_TOKEN đạt BLEU cao nhất khoảng 20,678 và BERTScore F1 cao nhất khoảng 0,900.

Tuy nhiên BGE_TOKEN đạt GPTScore cao nhất khoảng 4,291 và faithfulness cao nhất khoảng 4,730. BGE_STRUCTURE có ROUGE-L và relevance cao.

Vì vậy không nên kết luận JINA_TOKEN tốt nhất tuyệt đối. JINA_TOKEN giống gold answer hơn theo một số automatic metric, còn BGE_TOKEN được LLM Judge đánh giá tốt hơn về chất lượng tổng hợp và bám context.

### Giải thích thêm

Sự khác nhau giữa metric là bình thường. BLEU đo từ ngữ, còn judge đánh giá ý nghĩa và context. Đây là lý do nhóm không chọn pipeline chỉ bằng một cột.

---

## Slide: Generation Results – E5-Large

### Lời trình bày chính

Với E5-large, BGE_STRUCTURE có GPTScore cao nhất khoảng 4,217, correctness khoảng 3,993 và BERTScore khoảng 0,900.

JINA_STRUCTURE có BLEU khoảng 17,933 và ROUGE-L khoảng 0,408, nhỉnh hơn ở một số automatic metric. Tuy nhiên chênh lệch giữa các pipeline không quá lớn.

Điều này cho thấy khi retrieval đã tương đối tốt, answer quality còn phụ thuộc vào prompt, context order và cách Claude tổng hợp evidence, không chỉ phụ thuộc reranker.

---

## Slide: Generation Results – Qwen3

### Lời trình bày chính

Với Qwen3, JINA_TOKEN có BLEU cao nhất khoảng 20,843. Tuy nhiên các pipeline BGE có LLM Judge tốt hơn.

BGE_STRUCTURE có GPTScore cao nhất khoảng 4,070 và fluency cao nhất khoảng 4,644. BGE_TOKEN có faithfulness khoảng 4,461, cao nhất trong bảng.

Do đó cách kết luận chính xác là JINA_TOKEN tốt về overlap với gold answer, còn BGE pipeline tốt hơn theo judge score. Không nên nói BGE thắng ở toàn bộ automatic metric.

---

## Slide: Generation Results – BGE-M3

### Lời trình bày chính

Với BGE-M3, BGE_TOKEN đạt GPTScore cao nhất khoảng 4,404, relevance khoảng 4,461 và faithfulness khoảng 4,717.

JINA_TOKEN nhỉnh hơn nhẹ ở correctness khoảng 4,230 và completeness bằng 4,000.

Đây là nhóm có LLM Judge cao. Tuy nhiên BERTScore chỉ khoảng 0,66 đến 0,68, thấp bất thường so với khoảng 0,89 đến 0,90 ở các bảng khác. Nhóm cần kiểm tra lại model BERTScore, preprocessing và cách tổng hợp trước khi so sánh trực tiếp.

### Giải thích thêm

Khi một metric thay đổi mạnh trong khi các metric khác không thay đổi tương ứng, cần xem đây là dấu hiệu kiểm tra pipeline đánh giá, không nên ngay lập tức kết luận model kém.

---

## Slide: Best Current Pipeline

### Lời trình bày chính

Dựa trên kết quả retrieval hiện tại, cấu hình được đề xuất là Token chunking, E5-large embedding, JINA_TOKEN reranker và Claude generation.

Lý do thứ nhất là E5-large có dense retrieval mạnh về nDCG, MRR và Hit@1.

Lý do thứ hai là Token chunking cân bằng giữa context, số lượng chunk và index size.

Lý do thứ ba là E5 kết hợp JINA_TOKEN đạt Hit@1 bằng 0,842, MRR@5 bằng 0,882 và nDCG@5 bằng 0,809, cho thấy khả năng đưa evidence đúng lên đầu.

Tuy nhiên đây nên được gọi là best current retrieval pipeline, chưa phải best end-to-end pipeline tuyệt đối. Một số cấu hình khác có GPTScore cao hơn và các bảng vẫn cần chuẩn hóa. Kết luận cuối cùng phải xét retrieval quality, generation quality, latency, index size và faithfulness.

### Giải thích thêm

Nếu thầy cô hỏi vì sao không chọn BGE-M3 JINA_TOKEN khi nDCG@5 bằng 0,818, câu trả lời là: kết quả giữa các thành viên hoặc các lần chạy cần được tái kiểm chứng trên cùng benchmark và cùng cấu hình. E5 được chọn tạm thời vì dense retrieval ban đầu mạnh và ổn định, nhưng quyết định cuối kỳ vẫn mở.

---

# 3. Analysis, Plan & Final Contribution

## Slide: Qualitative Example

### Lời trình bày chính

Ngoài metric trung bình, nhóm cần phân tích từng ví dụ để hiểu hệ thống đúng hoặc sai ở đâu.

Mỗi ví dụ nên gồm question, gold answer, retrieved context title, generated answer, BERTScore, GPTScore và faithfulness. Nhóm dự kiến chọn hai ví dụ tốt và một ví dụ lỗi.

Slide hiện tại mới mô tả format phân tích, chưa phải kết quả qualitative hoàn chỉnh. Trước báo cáo cuối kỳ, nhóm sẽ thay nội dung kế hoạch bằng các ví dụ thực tế.

### Giải thích thêm

Ví dụ tốt cần chứng minh cả retrieval đúng và answer được context hỗ trợ. Ví dụ lỗi cần chỉ rõ nguyên nhân: không có evidence trong top-k, reranker đẩy evidence xuống, generator bỏ sót chi tiết hay gold answer có vấn đề.

---

## Slide: Error Analysis

### Lời trình bày chính

Nhóm chia lỗi thành bốn loại.

Retrieval error xảy ra khi top-k không chứa article hoặc chunk đúng. Nguyên nhân có thể là tên riêng, sự kiện hiếm, diễn đạt khác nhau hoặc ground truth chưa đầy đủ.

Chunking error xảy ra khi chunk quá ngắn làm mất thông tin trước sau, hoặc chunk quá dài chứa nhiều nội dung nhiễu.

Reranking error xảy ra khi evidence có trong candidate nhưng reranker đẩy xuống thấp, hoặc ưu tiên đoạn liên quan bề mặt mà không chứa đáp án.

Generation error xảy ra khi Claude trả lời thiếu chi tiết, diễn đạt quá chung, từ chối dù context đủ hoặc thêm thông tin ngoài context.

Phân loại lỗi giúp xác định thành phần cần sửa. Nếu evidence không có trong candidate thì cần cải thiện embedding hoặc chunking. Nếu evidence có nhưng rank thấp thì cần xem reranker. Nếu context đúng mà answer sai thì lỗi nằm ở generation hoặc prompt.

---

## Slide: Limitations

### Lời trình bày chính

Đề tài hiện có một số hạn chế.

Thứ nhất, QA set mới có 152 câu, chưa đủ lớn để khẳng định khả năng tổng quát trên toàn bộ tin tức tiếng Việt.

Thứ hai, QA do LLM sinh và chưa có human review hoàn chỉnh. Một số liên kết giữa QA và corpus, gold answer hoặc answerability vẫn cần rà soát.

Thứ ba, kết quả bốn embedding cần được tái kiểm chứng trên cùng cấu hình, cùng tập truy vấn và cùng phiên bản dữ liệu.

Thứ tư, BLEU và ROUGE không phản ánh đầy đủ semantic quality; LLM Judge có thể bias theo prompt hoặc model; BERTScore giữa một số bảng chưa nhất quán.

Thứ năm, nhóm chưa tối ưu sâu top-k, chunk size, overlap và rerank threshold; chưa có cost analysis chi tiết.

Cuối cùng, pipeline hiện mới là dense retrieval cộng reranking, chưa có BM25 hoặc hybrid retrieval.

### Giải thích thêm

Nêu limitation không làm giảm giá trị đề tài. Ngược lại, nó cho thấy nhóm hiểu phạm vi của kết quả và biết những gì cần kiểm chứng trước khi đưa ra kết luận cuối.

---

## Slide: 6-Week Plan

### Lời trình bày chính

Trong sáu tuần tiếp theo, nhóm dự kiến:

- Tuần một: chuẩn hóa benchmark, chốt QA version và thống nhất format metric.
- Tuần hai: làm ablation chunking trên Token, Structured và Recursive.
- Tuần ba: so sánh BGE reranker, Jina reranker và no-rerank.
- Tuần bốn: tối ưu prompt, context length và top-k.
- Tuần năm: human review từ 30 đến 50 mẫu, hoàn thiện error taxonomy và qualitative examples.
- Tuần sáu: đóng gói pipeline, hoàn thiện demo, báo cáo và slide cuối kỳ.

### Giải thích thêm

Ưu tiên cao nhất là chuẩn hóa benchmark. Nếu dữ liệu hoặc tập truy vấn khác nhau thì việc so sánh model không có ý nghĩa, dù bảng metric nhìn rất đầy đủ.

---

## Slide: Final Roadmap / Expected Contribution

### Lời trình bày chính

Đề tài kỳ vọng có ba đóng góp chính.

Thứ nhất là một benchmark thực nghiệm so sánh bốn dense embedding model cho retrieval trên tin tức tiếng Việt.

Thứ hai là phân tích ảnh hưởng của chunking và reranking đến thứ hạng evidence và chất lượng câu trả lời.

Thứ ba là một pipeline RAG có thể chạy lại từ xử lý dữ liệu, tạo chunk, embedding, retrieval, reranking, generation đến evaluation.

Best configuration cuối kỳ sẽ được lựa chọn dựa trên nhiều tiêu chí: retrieval quality, generation quality, latency, index size, faithfulness và khả năng xử lý unanswerable.

---

## Kết luận

### Lời trình bày chính

Tóm lại, ở giai đoạn giữa kỳ, nhóm đã xử lý corpus 10.073 bài, thực hiện EDA, xây dựng 152 QA, thử bốn chiến lược chunking, so sánh bốn dense embedding model, đánh giá BGE và Jina reranker, sinh câu trả lời bằng Claude và chấm bằng automatic metric cùng LLM Judge.

Kết quả hiện tại cho thấy Token chunking thường ổn định; E5-large mạnh về dense retrieval và rank đầu; BGE-M3 có sự cân bằng tốt về chất lượng và tài nguyên. E5-large kết hợp Token chunking và JINA_TOKEN là cấu hình retrieval nổi bật hiện tại, nhưng quyết định end-to-end cuối cùng vẫn cần benchmark thống nhất, human review và error analysis.

Trong giai đoạn tiếp theo, nhóm sẽ tập trung vào chuẩn hóa dữ liệu đánh giá, kiểm tra lại các metric chưa nhất quán, bổ sung no-rerank và hybrid baseline nếu có thời gian, tối ưu generation và hoàn thiện demo.

Em xin cảm ơn thầy cô và các bạn đã lắng nghe.

---

# PHẦN DỰ PHÒNG: CÂU HỎI THƯỜNG GẶP

## 1. Hệ thống có dùng Hybrid Retrieval không?

Chưa. Hệ thống hiện dùng dense semantic retrieval bằng cosine similarity, sau đó semantic reranking bằng BGE hoặc Jina. Hybrid retrieval đúng nghĩa cần có thêm lexical hoặc sparse retriever như BM25 và cơ chế fusion như RRF. README có định hướng hybrid nhưng source hiện tại chưa triển khai.

## 2. BGE-M3 có phải Hybrid không?

Không phải trong cách triển khai hiện tại. BGE-M3 có thể hỗ trợ nhiều chế độ biểu diễn trong một số framework, nhưng code của đồ án dùng SentenceTransformer.encode để tạo một dense vector và truy xuất bằng cosine similarity. Vì vậy thí nghiệm hiện tại vẫn là dense retrieval.

## 3. Embedding model và reranker khác nhau thế nào?

Embedding model biến từng query và chunk thành vector độc lập, cho phép so sánh nhanh với toàn corpus. Reranker nhận trực tiếp một cặp query–chunk và chấm mức liên quan chính xác hơn nhưng chậm hơn. Vì vậy embedding dùng để lấy candidate, reranker dùng để sắp xếp lại candidate.

## 4. Vì sao chọn Token chunking?

Token tạo khoảng 21 nghìn chunk với độ dài trung bình khoảng 365 token, đủ context nhưng index không quá lớn. Structured có kết quả gần tương tự nhưng nhiều chunk hơn. Recursive tạo rất nhiều chunk ngắn, tăng index và có nguy cơ thiếu context.

## 5. Vì sao E5-large được đề xuất?

E5-large có dense retrieval mạnh trên Token và Structured, đặc biệt nDCG, MRR và Hit@1. Khi kết hợp Token với Jina reranker, nó đạt Hit@1 0,842 và MRR@5 0,882. Tuy nhiên đây là đề xuất retrieval tạm thời, chưa phải kết luận end-to-end cuối cùng.

## 6. Vì sao không chỉ dùng BLEU?

BLEU đo trùng n-gram. Một câu trả lời đúng nghĩa nhưng diễn đạt khác gold answer có thể có BLEU thấp. Vì vậy nhóm dùng thêm ROUGE-L, BERTScore và LLM Judge. Faithfulness đặc biệt quan trọng vì câu trả lời phải được context hỗ trợ.

## 7. Human review đã hoàn thành chưa?

Chưa hoàn thành đầy đủ. QA hiện đã được tạo và validation tự động, nhưng cần duyệt thủ công trước khi chốt benchmark. Trong slide nên dùng cách nói “human review cần hoàn thiện”, không nói “toàn bộ QA đã được con người kiểm tra”.

## 8. Vì sao retrieval có thể tốt nhưng generation chưa tốt?

Retriever chỉ cung cấp context. Generator vẫn có thể bỏ sót thông tin, chọn nhầm giữa nhiều context, diễn đạt quá chung hoặc không tuân thủ prompt. Ngoài ra gold answer và metric cũng có thể ảnh hưởng điểm. Vì vậy cần đánh giá riêng retrieval và generation.

## 9. Vì sao BERTScore của BGE-M3 thấp bất thường?

Hiện chưa thể kết luận nguyên nhân. Có thể do khác model BERTScore, preprocessing, normalization hoặc script tổng hợp. Đây là dấu hiệu cần kiểm tra lại pipeline đánh giá trước khi so sánh trực tiếp với các bảng khác.

## 10. Hạn chế lớn nhất hiện tại là gì?

Hạn chế lớn nhất là benchmark nhỏ và chưa human review hoàn chỉnh, cùng với việc các kết quả do nhiều cấu hình hoặc nhiều người chạy cần được chuẩn hóa trên cùng dữ liệu. Nếu benchmark chưa ổn định thì kết luận model tốt nhất chưa đủ chắc chắn.
