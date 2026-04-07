#!/usr/bin/env python

from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys
from argparse import ArgumentParser
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LANDMARK5_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

IMAGE_TAG_PATTERN = re.compile(
    r'<img\b(?P<before>[^>]*)\bsrc=(?P<quote>["\'])(?P<src>.+?)(?P=quote)(?P<after>[^>]*)>',
    re.IGNORECASE,
)


def load_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        print('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu')
        sys.exit(1)
    return ort


def load_skimage_transform():
    try:
        from skimage import transform as trans  # type: ignore
    except ImportError:
        print('ERROR: scikit-image is required for landmark-based alignment. pip install scikit-image')
        sys.exit(1)
    return trans


def load_markdown_renderer():
    try:
        from markdown_it import MarkdownIt  # type: ignore
    except ImportError:
        print('ERROR: markdown-it-py is required for markdown to HTML conversion. pip install markdown-it-py')
        sys.exit(1)
    return MarkdownIt('commonmark').enable('table')


def build_providers(ort, execution_provider: str) -> List[str]:
    available_providers = ort.get_available_providers()
    if execution_provider == 'cpu':
        return ['CPUExecutionProvider']
    if execution_provider == 'cuda':
        if 'CUDAExecutionProvider' not in available_providers:
            print('ERROR: CUDAExecutionProvider is not available in this environment.')
            print(f'Available providers: {available_providers}')
            sys.exit(1)
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f'ERROR: Unsupported execution provider: {execution_provider}')
    sys.exit(1)


def normalize_landmark_key(path: str) -> str:
    return os.path.normpath(path.replace('\\', '/'))


def load_landmarks_file(landmarks_file: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not os.path.isfile(landmarks_file):
        print(f'ERROR: Landmarks file does not exist: {landmarks_file}')
        sys.exit(1)

    landmarks_by_key: Dict[str, np.ndarray] = {}
    landmarks_by_abs: Dict[str, np.ndarray] = {}
    with open(landmarks_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            parts = stripped_line.split()
            if len(parts) < 12:
                print(f'ERROR: Invalid landmarks line at {landmarks_file}:{line_number}')
                print('Expected format: image_path x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 score')
                sys.exit(1)

            landmark_key = normalize_landmark_key(parts[0])
            if landmark_key in landmarks_by_key:
                print(f'ERROR: Duplicate landmark path in file: {parts[0]}')
                sys.exit(1)

            try:
                landmark5 = np.array([float(value) for value in parts[1:11]], dtype=np.float32).reshape((5, 2))
                float(parts[-1])
            except ValueError:
                print(f'ERROR: Failed to parse landmark coordinates at {landmarks_file}:{line_number}')
                sys.exit(1)

            landmarks_by_key[landmark_key] = landmark5

            absolute_landmark_key = os.path.abspath(landmark_key)
            if absolute_landmark_key in landmarks_by_abs:
                print(f'ERROR: Duplicate normalized absolute landmark path in file: {parts[0]}')
                sys.exit(1)
            landmarks_by_abs[absolute_landmark_key] = landmark5

    return landmarks_by_key, landmarks_by_abs


def resolve_landmark5(
    image_path: str,
    landmarks_by_key: Optional[Dict[str, np.ndarray]] = None,
    landmarks_by_abs: Optional[Dict[str, np.ndarray]] = None,
    images_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    if not landmarks_by_key and not landmarks_by_abs:
        return None

    if images_dir is not None:
        relative_path = os.path.relpath(image_path, start=images_dir)
        return landmarks_by_key.get(normalize_landmark_key(relative_path))

    normalized_image_path = normalize_landmark_key(image_path)
    landmark5 = landmarks_by_key.get(normalized_image_path)
    if landmark5 is not None:
        return landmark5

    absolute_image_path = os.path.abspath(normalized_image_path)
    return landmarks_by_abs.get(absolute_image_path)


def load_image_as_input(
    image_path: str,
    landmarks_by_key: Optional[Dict[str, np.ndarray]] = None,
    landmarks_by_abs: Optional[Dict[str, np.ndarray]] = None,
    images_dir: Optional[str] = None,
) -> Tuple[np.ndarray, bool]:
    image = cv2.imread(image_path)
    if image is None:
        print(f'ERROR: Failed to read image: {image_path}')
        sys.exit(1)

    landmark5 = resolve_landmark5(
        image_path=image_path,
        landmarks_by_key=landmarks_by_key,
        landmarks_by_abs=landmarks_by_abs,
        images_dir=images_dir,
    )
    aligned = landmark5 is not None
    if landmark5 is not None:
        trans = load_skimage_transform()
        transform = trans.SimilarityTransform()
        if not transform.estimate(landmark5, LANDMARK5_SRC):
            print(f'ERROR: Failed to estimate similarity transform for image: {image_path}')
            sys.exit(1)
        image = cv2.warpAffine(image, transform.params[0:2, :], (112, 112), borderValue=0.0)
    else:
        image = cv2.resize(image, (112, 112))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image / 255.0
    image = (image - 0.5) / 0.5
    return image, aligned


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def list_image_files(dir_path: str) -> List[str]:
    path = Path(dir_path)
    image_files = []
    for extension in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(path.rglob(extension))
    return sorted(str(file) for file in image_files)


def is_dynamic_dim(dim: object) -> bool:
    return dim is None or isinstance(dim, str)


def choose_batch_size(input_shape: List[object], output_shape: List[object]) -> int:
    input_batch_dim = input_shape[0] if input_shape else None
    output_batch_dim = output_shape[0] if output_shape else None
    if input_batch_dim == 1 or output_batch_dim == 1:
        return 1
    return 32


def load_session(model_path: str, execution_provider: str):
    if not os.path.isfile(model_path):
        print(f'ERROR: Model file does not exist: {model_path}')
        sys.exit(1)

    ort = load_onnxruntime()
    providers = build_providers(ort, execution_provider)

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(f'ERROR: Failed to load ONNX model: {model_path}')
        print(exc)
        sys.exit(1)

    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1:
        print(f'ERROR: Expected exactly 1 input, but got {len(input_meta)}')
        sys.exit(1)
    if len(output_meta) != 1:
        print(f'ERROR: Expected exactly 1 output, but got {len(output_meta)}')
        sys.exit(1)

    input_tensor = input_meta[0]
    output_tensor = output_meta[0]
    input_shape = list(input_tensor.shape)
    output_shape = list(output_tensor.shape)
    if len(input_shape) != 4 or input_shape[1:] != [3, 112, 112]:
        print(f'ERROR: Unexpected input shape: {input_tensor.shape}, expected: [N, 3, 112, 112] or [1, 3, 112, 112]')
        sys.exit(1)
    if len(output_shape) != 2 or output_shape[1] != 512:
        print(f'ERROR: Unexpected output shape: {output_tensor.shape}, expected: [N, 512] or [1, 512]')
        sys.exit(1)

    input_batch_dim = input_shape[0]
    output_batch_dim = output_shape[0]
    if not is_dynamic_dim(input_batch_dim) and not is_dynamic_dim(output_batch_dim):
        if int(input_batch_dim) != int(output_batch_dim):
            print(
                'ERROR: Input/output batch dimensions do not match: '
                f'input={input_tensor.shape}, output={output_tensor.shape}'
            )
            sys.exit(1)

    batch_size = choose_batch_size(input_shape, output_shape)

    if not is_dynamic_dim(input_batch_dim) and int(input_batch_dim) not in [1, batch_size]:
        print(f'ERROR: Unsupported fixed input batch dimension: {input_batch_dim}')
        sys.exit(1)
    if input_tensor.type != 'tensor(float)':
        print(f'ERROR: Unexpected input type: {input_tensor.type}, expected: tensor(float)')
        sys.exit(1)
    if output_tensor.type != 'tensor(float)':
        print(f'ERROR: Unexpected output type: {output_tensor.type}, expected: tensor(float)')
        sys.exit(1)

    return session, input_tensor, output_tensor, batch_size


def run_inference(session, input_tensor, output_tensor, batch: np.ndarray) -> np.ndarray:
    try:
        features = session.run([output_tensor.name], {input_tensor.name: batch})[0]
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    return features


def encode_images(
    session,
    input_tensor,
    output_tensor,
    image_paths: List[str],
    landmarks_by_key: Optional[Dict[str, np.ndarray]] = None,
    landmarks_by_abs: Optional[Dict[str, np.ndarray]] = None,
    images_dir: Optional[str] = None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, int]:
    all_features = []
    aligned_count = 0
    for start_index in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start_index:start_index + batch_size]
        batch_images = []
        for image_path in batch_paths:
            batch_image, aligned = load_image_as_input(
                image_path=image_path,
                landmarks_by_key=landmarks_by_key,
                landmarks_by_abs=landmarks_by_abs,
                images_dir=images_dir,
            )
            batch_images.append(batch_image)
            aligned_count += int(aligned)
        batch = np.stack(batch_images, axis=0).astype(np.float32)
        try:
            batch_features = run_inference(session, input_tensor, output_tensor, batch)
        except RuntimeError as exc:
            if len(batch_paths) == 1:
                print('ERROR: ONNX inference failed.')
                print(exc)
                sys.exit(1)

            batch_features_list = []
            for batch_image in batch_images:
                try:
                    single_batch = np.expand_dims(batch_image, axis=0).astype(np.float32)
                    single_features = run_inference(session, input_tensor, output_tensor, single_batch)
                except RuntimeError as single_exc:
                    print('ERROR: ONNX inference failed.')
                    print(single_exc)
                    sys.exit(1)
                if single_features.shape != (1, 512):
                    print(
                        'ERROR: Unexpected feature output shape during single-image fallback: '
                        f'{single_features.shape}, expected: (1, 512)'
                    )
                    sys.exit(1)
                batch_features_list.append(single_features[0])
            batch_features = np.stack(batch_features_list, axis=0)
        if batch_features.shape != (len(batch_paths), 512):
            print(
                'ERROR: Unexpected feature output shape: '
                f'{batch_features.shape}, expected: ({len(batch_paths)}, 512)'
            )
            sys.exit(1)
        all_features.append(batch_features)
    return np.concatenate(all_features, axis=0), aligned_count


def write_markdown_results(
    output_path: str,
    model_path: str,
    images_dir: str,
    image_paths: List[str],
    normalized_features: np.ndarray,
    landmarks_file: Optional[str],
    aligned_count: int,
) -> int:
    rows: List[Tuple[str, str, str, str, float]] = []
    markdown_dir = os.path.dirname(os.path.abspath(output_path))
    for index1 in range(len(image_paths)):
        for index2 in range(index1 + 1, len(image_paths)):
            relative_path1 = os.path.relpath(image_paths[index1], start=images_dir)
            relative_path2 = os.path.relpath(image_paths[index2], start=images_dir)
            markdown_image_path1 = os.path.relpath(image_paths[index1], start=markdown_dir).replace(os.sep, '/')
            markdown_image_path2 = os.path.relpath(image_paths[index2], start=markdown_dir).replace(os.sep, '/')
            cosine_similarity = float(np.sum(normalized_features[index1] * normalized_features[index2]))
            rows.append((relative_path1, markdown_image_path1, relative_path2, markdown_image_path2, cosine_similarity))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('# Cosine Similarity Results\n\n')
        file.write(f'- Model: `{model_path}`\n')
        file.write(f'- Images directory: `{images_dir}`\n')
        file.write(f'- Landmarks file: `{landmarks_file}`\n' if landmarks_file is not None else '- Landmarks file: None\n')
        file.write(f'- Image count: {len(image_paths)}\n')
        file.write(f'- Aligned images: {aligned_count}/{len(image_paths)}\n')
        file.write(f'- Pair count: {len(rows)}\n\n')
        file.write('| Image 1 Path | Image 1 | Image 2 Path | Image 2 | Cosine Similarity |\n')
        file.write('| --- | --- | --- | --- | ---: |\n')
        for relative_path1, markdown_image_path1, relative_path2, markdown_image_path2, cosine_similarity in rows:
            file.write(
                f'| `{relative_path1}` | <img src="{markdown_image_path1}" width="112"> '
                f'| `{relative_path2}` | <img src="{markdown_image_path2}" width="112"> '
                f'| {cosine_similarity:.6f} |\n'
            )
    return len(rows)


def extract_html_title(markdown_text: str, input_markdown: str) -> str:
    for line in markdown_text.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('# '):
            return stripped_line[2:].strip()
    return Path(input_markdown).stem


def resolve_output_html_path(input_markdown: str, output_html: Optional[str]) -> str:
    if output_html is not None:
        return output_html
    return str(Path(input_markdown).with_suffix('.html'))


def build_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None or not mime_type.startswith('image/'):
        print(f'ERROR: Failed to determine image MIME type: {image_path}')
        sys.exit(1)

    encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def embed_html_images(html_text: str, markdown_dir: Path, embedded_images: Dict[str, str]) -> str:
    def replace_image(match: re.Match[str]) -> str:
        image_src = match.group('src')
        if '://' in image_src or image_src.startswith('//') or image_src.startswith('data:'):
            print(f'ERROR: Unsupported non-local image source in markdown: {image_src}')
            sys.exit(1)

        image_path = (markdown_dir / image_src).resolve()
        if not image_path.is_file():
            print(f'ERROR: Referenced image does not exist: {image_src}')
            print(f'Resolved path: {image_path}')
            sys.exit(1)

        embedded_src = embedded_images.get(image_src)
        if embedded_src is None:
            embedded_src = build_data_uri(image_path)
            embedded_images[image_src] = embedded_src
        return (
            f'<img{match.group("before")}src={match.group("quote")}{embedded_src}'
            f'{match.group("quote")}{match.group("after")}>'
        )

    return IMAGE_TAG_PATTERN.sub(replace_image, html_text)


def parse_markdown_table_row(line: str) -> List[str]:
    stripped_line = line.strip()
    if not stripped_line.startswith('|') or not stripped_line.endswith('|'):
        print(f'ERROR: Invalid markdown table row: {line}')
        sys.exit(1)
    return [cell.strip() for cell in stripped_line[1:-1].split('|')]


def split_markdown_table_sections(lines: List[str]) -> Tuple[List[str], Optional[List[str]], List[str]]:
    for index in range(len(lines) - 1):
        if lines[index].strip().startswith('|') and lines[index + 1].strip().startswith('|'):
            return lines[:index], lines[index:index + 2], lines[index + 2:]
    return lines, None, []


def write_html_document_start(output_file, title: str) -> None:
    output_file.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    body {{
      margin: 0;
      background: #f5f7fb;
      color: #162033;
    }}
    main {{
      box-sizing: border-box;
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2, h3 {{
      line-height: 1.2;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d7deea;
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #eaf0f8;
    }}
    code {{
      background: #eef3fa;
      border-radius: 4px;
      padding: 1px 4px;
    }}
    img {{
      display: block;
      max-width: 112px;
      height: auto;
    }}
  </style>
</head>
<body>
  <main>
""")


def write_html_document_end(output_file) -> None:
    output_file.write("""  </main>
</body>
</html>
""")


def write_markdown_table_html(
    output_file,
    table_header_lines: List[str],
    table_body_lines: List[str],
    renderer,
    markdown_dir: Path,
    embedded_images: Dict[str, str],
) -> None:
    header_cells = parse_markdown_table_row(table_header_lines[0])
    output_file.write('<table>\n<thead>\n<tr>\n')
    for cell in header_cells:
        rendered_cell = embed_html_images(renderer.renderInline(cell).strip(), markdown_dir, embedded_images)
        output_file.write(f'<th>{rendered_cell}</th>\n')
    output_file.write('</tr>\n</thead>\n<tbody>\n')

    for line in table_body_lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        cells = parse_markdown_table_row(line)
        output_file.write('<tr>\n')
        for cell in cells:
            rendered_cell = embed_html_images(renderer.renderInline(cell).strip(), markdown_dir, embedded_images)
            output_file.write(f'<td>{rendered_cell}</td>\n')
        output_file.write('</tr>\n')
    output_file.write('</tbody>\n</table>\n')


def convert_markdown_to_html(input_markdown: str, output_html: Optional[str]) -> str:
    input_path = Path(input_markdown)
    if not input_path.is_file():
        print(f'ERROR: Input markdown does not exist: {input_markdown}')
        sys.exit(1)

    output_path = Path(resolve_output_html_path(input_markdown, output_html))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_text = input_path.read_text(encoding='utf-8')
    title = extract_html_title(markdown_text, input_markdown)
    renderer = load_markdown_renderer()
    markdown_dir = input_path.resolve().parent
    embedded_images: Dict[str, str] = {}
    lines = markdown_text.splitlines()
    preamble_lines, table_header_lines, table_body_lines = split_markdown_table_sections(lines)

    with output_path.open('w', encoding='utf-8') as output_file:
        write_html_document_start(output_file, title)

        preamble_markdown = '\n'.join(preamble_lines).strip()
        if preamble_markdown:
            rendered_preamble = renderer.render(preamble_markdown + '\n')
            output_file.write(embed_html_images(rendered_preamble, markdown_dir, embedded_images))

        if table_header_lines is not None:
            write_markdown_table_html(
                output_file=output_file,
                table_header_lines=table_header_lines,
                table_body_lines=table_body_lines,
                renderer=renderer,
                markdown_dir=markdown_dir,
                embedded_images=embedded_images,
            )

        write_html_document_end(output_file)

    return str(output_path)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='ONNX model path for TransFace.',
    )
    parser.add_argument(
        '--input_markdown',
        type=str,
        default=None,
        help='Path to an existing markdown results file to convert into a self-contained HTML file.',
    )
    parser.add_argument(
        '--output_html',
        type=str,
        default=None,
        help='Output HTML path for markdown conversion mode. Defaults to the input markdown path with an .html suffix.',
    )
    parser.add_argument(
        '--image1',
        type=str,
        default=None,
        help='Path to the first aligned head/face image.',
    )
    parser.add_argument(
        '--image2',
        type=str,
        default=None,
        help='Path to the second aligned head/face image.',
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default=None,
        help='Directory containing aligned head/face images to compare recursively.',
    )
    parser.add_argument(
        '--output_markdown',
        type=str,
        default='results.md',
        help='Output markdown path for folder comparison mode.',
    )
    parser.add_argument(
        '--landmarks_file',
        type=str,
        default=None,
        help='IJB-format TXT file containing 5-point landmarks.',
    )
    parser.add_argument(
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Execution provider for ONNXRuntime.',
    )
    args = parser.parse_args()

    inference_args = [args.model, args.image1, args.image2, args.images_dir]
    if args.input_markdown is not None:
        if any(value is not None for value in inference_args):
            print('ERROR: --input_markdown cannot be combined with inference arguments.')
            sys.exit(1)
        output_html = convert_markdown_to_html(args.input_markdown, args.output_html)
        print(f'Input markdown: {args.input_markdown}')
        print(f'HTML output: {output_html}')
        return

    if args.output_html is not None:
        print('ERROR: --output_html requires --input_markdown.')
        sys.exit(1)

    if args.model is None:
        print('ERROR: --model is required unless --input_markdown is specified.')
        sys.exit(1)

    if args.images_dir is not None and (args.image1 is not None or args.image2 is not None):
        print('ERROR: Specify either --images_dir or both --image1 and --image2, not both modes together.')
        sys.exit(1)
    if args.images_dir is None and (args.image1 is None or args.image2 is None):
        print('ERROR: Specify either --images_dir or both --image1 and --image2.')
        sys.exit(1)

    session, input_tensor, output_tensor, batch_size = load_session(args.model, args.execution_provider)
    landmarks_by_key: Dict[str, np.ndarray] = {}
    landmarks_by_abs: Dict[str, np.ndarray] = {}
    if args.landmarks_file is not None:
        landmarks_by_key, landmarks_by_abs = load_landmarks_file(args.landmarks_file)

    if args.images_dir is not None:
        if not os.path.isdir(args.images_dir):
            print(f'ERROR: Images directory does not exist: {args.images_dir}')
            sys.exit(1)

        image_paths = list_image_files(args.images_dir)
        if len(image_paths) < 2:
            print(f'ERROR: At least 2 images are required in folder mode, found: {len(image_paths)}')
            sys.exit(1)

        features, aligned_count = encode_images(
            session,
            input_tensor,
            output_tensor,
            image_paths,
            landmarks_by_key=landmarks_by_key,
            landmarks_by_abs=landmarks_by_abs,
            images_dir=args.images_dir,
            batch_size=batch_size,
        )
        normalized_features = l2_normalize(features)
        rows_count = write_markdown_results(
            output_path=args.output_markdown,
            model_path=args.model,
            images_dir=args.images_dir,
            image_paths=image_paths,
            normalized_features=normalized_features,
            landmarks_file=args.landmarks_file,
            aligned_count=aligned_count,
        )

        print(f'Model: {args.model}')
        print(f'Images directory: {args.images_dir}')
        print(f'Landmarks file: {args.landmarks_file}' if args.landmarks_file is not None else 'Landmarks file: None')
        print(f'Image count: {len(image_paths)}')
        print(f'Aligned images: {aligned_count}/{len(image_paths)}')
        print(f'Embedding shape: {features.shape}')
        print(f'Pair count: {rows_count}')
        print(f'Markdown output: {args.output_markdown}')
        return

    features, aligned_count = encode_images(
        session,
        input_tensor,
        output_tensor,
        [args.image1, args.image2],
        landmarks_by_key=landmarks_by_key,
        landmarks_by_abs=landmarks_by_abs,
        batch_size=batch_size,
    )
    if features.shape != (2, 512):
        print(f'ERROR: Unexpected feature output shape: {features.shape}, expected: (2, 512)')
        sys.exit(1)

    normalized_features = l2_normalize(features)
    cosine_similarity = float(np.sum(normalized_features[0] * normalized_features[1]))

    print(f'Model: {args.model}')
    print(f'Image1: {args.image1}')
    print(f'Image2: {args.image2}')
    print(f'Landmarks file: {args.landmarks_file}' if args.landmarks_file is not None else 'Landmarks file: None')
    print(f'Aligned images: {aligned_count}/2')
    print(f'Embedding shape: {features.shape}')
    print(f'Cosine similarity: {cosine_similarity:.6f}')


if __name__ == '__main__':
    main()
