from pathlib import Path

import cv2
import pytesseract


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "img" / "51662333_Front.jpg"
TESSERACT_CANDIDATES = [
	Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
	Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


def configure_tesseract() -> Path:
	for candidate in TESSERACT_CANDIDATES:
		if candidate.exists():
			pytesseract.pytesseract.tesseract_cmd = str(candidate)
			return candidate
	raise FileNotFoundError("No se encontro tesseract.exe en las rutas esperadas.")



def available_language() -> str:
	try:
		languages = set(pytesseract.get_languages(config=""))
	except pytesseract.TesseractError:
		return "eng"
	return "spa" if "spa" in languages else "eng"


def preprocess_image(image_path: Path):
	image = cv2.imread(str(image_path))
	if image is None:
		raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

	height, width = image.shape[:2]
	roi = image[0:height, 0:int(width * 0.6)]

	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	
	# Normalizar el histograma para imágenes muy claras u oscuras
	normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
	
	# Mejorar contraste adaptativo
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	contrasted = clahe.apply(normalized)
	
	# Filtro bilateral para reducir ruido manteniendo bordes
	filtered = cv2.bilateralFilter(contrasted, 11, 80, 80)
	
	# Escalar antes de binarizar para mejor precisión
	scaled = cv2.resize(filtered, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	
	# Combinar OTSU con threshold adaptativo para mayor robustez
	_, otsu_thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	adaptive_thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
											cv2.THRESH_BINARY, 15, 10)
	
	# Usar la combinación que produzca menos ruido
	otsu_noise = cv2.countNonZero(cv2.bitwise_not(otsu_thresh)) / (scaled.shape[0] * scaled.shape[1])
	adaptive_noise = cv2.countNonZero(cv2.bitwise_not(adaptive_thresh)) / (scaled.shape[0] * scaled.shape[1])
	
	processed = otsu_thresh if otsu_noise < adaptive_noise else adaptive_thresh
	
	# Limpieza final con operaciones morfológicas
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
	
	return processed


def main() -> None:
	tesseract_path = configure_tesseract()
	language = available_language()
	processed_image = preprocess_image(IMAGE_PATH)
	config = "--oem 3 --psm 6"
	text = pytesseract.image_to_string(processed_image, lang=language, config=config)

	print(f"Imagen: {IMAGE_PATH}")
	print(f"Ruta de Tesseract: {tesseract_path}")
	print(f"Idioma OCR usado: {language}")
	print("Texto extraido:")
	print(text.strip())


if __name__ == "__main__":
	main()
