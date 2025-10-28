# 建筑图纸问答 RAG 系统

本系统基于 RAG（检索增强生成）架构，支持对建筑/施工图纸的自然语言问答。

## 概述

本系统可处理大型 PDF 建筑图纸（约60MB，20-30页），支持对平面图、图例、标题栏、规范等内容的智能问答。采用语义检索（FAISS）+关键词检索（BM25）混合方案，并支持智能 OCR 回退。

**主要特性：**
- 混合检索（FAISS 语义 + BM25 关键词）
- 图形内容智能 OCR 回退
- 答案带引用（页码、区块、bbox）
- 零幻觉（不确定时返回 "no evidence found"）
- 响应快（~2秒/问）
- 本地 CPU 运行，无需 GPU/云

---

## 快速开始

### 环境依赖
- Python 3.8+
- Tesseract OCR
- 建议 16GB 内存

### 安装
```bash
# 克隆仓库
git clone git@github.com:ishamishra0408/RAG-for-Design-Drawings.git
cd rag_drawings

# 安装依赖
pip install -r requirements.txt

# 安装 Tesseract OCR
# macOS:
brew install tesseract
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# Windows:
# 参考 https://github.com/UB-Mannheim/tesseract/wiki
```

### 配置与 CLI 命令顺序

#### 1. 放置 PDF 文件
将 PDF 放入 `data/` 目录，并在 `config.yaml` 设置路径：
```yaml
pdf_path: "./data/your_drawing.pdf"
```

#### 2. 依次执行以下命令

**（1）数据预处理**
```bash
python app.py ingest
```
> 解析 PDF，生成数据块（chunks.jsonl）

**（2）构建索引**
```bash
python app.py index
```
> 基于数据块构建 FAISS/BM25 索引

**（3）问答示例**
```bash
python app.py ask "项目地址是什么？"
python app.py ask "建筑师是谁？"
python app.py ask "用地面积是多少？"
```
> 支持自然语言提问，返回答案、置信度和引用

**（4）批量演示（可选）**
```bash
python app.py demo
```
> 批量演示内置问题

#### 典型输出格式
```json
{
  "answer": "PROJECTADDRESS: 23KERLEYCOURT",
  "confidence": 0.85,
  "citations": [
    {
      "sheet_id": "S1",
      "page": 1,
      "section": "title block",
      "bbox": "..."
    }
  ]
}
```

---

## 其它说明
- 支持英文/中文自然语言提问
- 若返回 "no evidence found"，说明数据中未检索到相关内容或置信度不足
- 详细架构、技术栈、评测等请参考英文版 README
