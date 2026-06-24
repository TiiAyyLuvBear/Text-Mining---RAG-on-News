# Embedding Strategy Samples

Report này so sánh text/metadata được đưa vào embedding giữa các chunking strategy cho cùng article.

## Article `10000`

### `token`

- Source: `src\chunking\output\vieonline_news_chunks_token.jsonl`
- Num chunks: 2

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `internal_token_window` | `content` | 450 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 1 | `internal_token_window` | `content` | 352 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |

### `langchain_recursive`

- Source: `src\chunking\output\vieonline_news_chunks_langchain_recursive.jsonl`
- Num chunks: 10

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 83 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 1 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 87 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 2 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 75 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 3 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 89 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 4 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 50 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |

### `llamaindex`

- Source: `src\chunking\output\vieonline_news_chunks_llamaindex.jsonl`
- Num chunks: 4

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 198 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 1 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 199 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 2 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 210 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 3 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 167 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |

### `structured`

- Source: `src\chunking\output\vieonline_news_chunks_structured.jsonl`
- Num chunks: 2

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `internal_structured_sentence_window` | `content` | 424 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |
| 1 | `internal_structured_sentence_window` | `content` | 391 | Tiêu đề: Tom Cruise vẫn 'đặt cược' sinh mạng trong Mission: Impossible 8 Mô tả: Siêu sao hành động Tom Cruise vẫn miệt mài 'vào sinh ra tử' cùng vai diễn, suýt chết nhiều lần khi đóng phim Mission: Impossible - The Fi... |

## Article `100056`

### `token`

- Source: `src\chunking\output\vieonline_news_chunks_token.jsonl`
- Num chunks: 5

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `internal_token_window` | `content` | 450 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 1 | `internal_token_window` | `content` | 450 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 2 | `internal_token_window` | `content` | 450 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 3 | `internal_token_window` | `content` | 450 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 4 | `internal_token_window` | `content` | 170 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |

### `langchain_recursive`

- Source: `src\chunking\output\vieonline_news_chunks_langchain_recursive.jsonl`
- Num chunks: 20

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 65 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 1 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 90 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 2 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 84 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 3 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 74 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 4 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 86 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |

### `llamaindex`

- Source: `src\chunking\output\vieonline_news_chunks_llamaindex.jsonl`
- Num chunks: 9

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 173 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 1 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 184 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 2 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 192 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 3 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 202 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 4 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 206 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |

### `structured`

- Source: `src\chunking\output\vieonline_news_chunks_structured.jsonl`
- Num chunks: 5

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `internal_structured_sentence_window` | `content` | 418 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 1 | `internal_structured_sentence_window` | `content` | 439 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 2 | `internal_structured_sentence_window` | `content` | 419 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 3 | `internal_structured_sentence_window` | `content` | 426 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |
| 4 | `internal_structured_sentence_window` | `content` | 374 | Tiêu đề: Sắp xếp, sáp nhập trường ĐH: Quyền lợi của người lao động, sinh viên ra sao? Mô tả: Theo các chuyên gia, nhiều vấn đề đặt ra cần giải quyết khi xây dựng đề án sáp nhập các trường đại học như: nhân sự, cơ sở v... |

## Article `10007`

### `token`

- Source: `src\chunking\output\vieonline_news_chunks_token.jsonl`
- Num chunks: 2

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `internal_token_window` | `content` | 450 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: Những phác thảo ban đầu của d... |
| 1 | `internal_token_window` | `content` | 199 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: đám trên toàn thế giới, với 2... |

### `langchain_recursive`

- Source: `src\chunking\output\vieonline_news_chunks_langchain_recursive.jsonl`
- Num chunks: 8

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 64 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: Những phác thảo ban đầu của d... |
| 1 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 56 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: . Đây là một trong những sự k... |
| 2 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 48 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: . Tuy chưa có nhiều chi tiết... |
| 3 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 89 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: . Universal Century là điểm k... |
| 4 | `langchain_text_splitters.RecursiveCharacterTextSplitter` | `content` | 76 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: . Legendary Entertainmentcũng... |

### `llamaindex`

- Source: `src\chunking\output\vieonline_news_chunks_llamaindex.jsonl`
- Num chunks: 4

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 166 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: Những phác thảo ban đầu của d... |
| 1 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 121 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: Universal Century là điểm khở... |
| 2 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 175 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: Legendary Entertainmentcũng t... |
| 3 | `llama_index.core.node_parser.SentenceSplitter` | `content` | 140 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: Tất cả đều khởi đầu từ animeM... |

### `structured`

- Source: `src\chunking\output\vieonline_news_chunks_structured.jsonl`
- Num chunks: 2

| Chunk | Implementation | Structure | Tokens | Preview |
| ---: | --- | --- | ---: | --- |
| 0 | `internal_structured_sentence_window` | `content` | 429 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: Những phác thảo ban đầu của d... |
| 1 | `internal_structured_sentence_window` | `content` | 241 | Tiêu đề: Bandai quyết tâm đưa Gundam lên màn ảnh rộng Mô tả: Sau nhiều năm gián đoạn, dự án phim Gundam live-action cuối cùng cũng chính thức sản xuất. Chuyên mục: Giải trí Đoạn nội dung: RX-78 hay còn được biết đến n... |


