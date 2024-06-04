from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import UploadedImage
from .serializers import UploadedImageSerializer
from django.core.files.base import ContentFile
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import torch.nn.functional as F
import io
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


@method_decorator(csrf_exempt, name='dispatch')
class MlModelAPIView(APIView):
    def post(self, request):
        file = request.FILES.get('image')
        if not file:
            print("No image provided")
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        confidence_threshold = request.data.get('confidence_threshold', 0.5)
        try:
            confidence_threshold = float(confidence_threshold)
        except ValueError:
            return Response({"error": "Invalid confidence threshold"}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_image = UploadedImage(image=file)
        uploaded_image.save()

        image_path = uploaded_image.image.  path
        yolo_model_path = 'core/detect.pt'
        classification_model_path = 'core/class.pth'
        processed_image = self.process_image(image_path, yolo_model_path, classification_model_path,
                                             confidence_threshold)

        processed_image_io = io.BytesIO()
        processed_image.save(processed_image_io, format='JPEG')
        uploaded_image.processed_image.save(f"processed_{uploaded_image.image.name}",
                                            ContentFile(processed_image_io.getvalue()), save=True)

        print("Processed image saved")

        serializer = UploadedImageSerializer(uploaded_image)
        print("Serializer data:", serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def process_image(self, image_path, yolo_model_path, classification_model_path, confidence_threshold):

        def prediction(image_path, yolo_model_path, classification_model_path, confidence_threshold,
                       colors=[(0, 255, 0), (255, 0, 0)], thickness=3, scale=1.0,
                       show_names=True, show_confidence=True, num_classes=2,
                       text_size=15, text_color=(255, 255, 255), show_probabilities=True):
            print("Prediction function started")
            print("Confidence threshold:", confidence_threshold)

            def load_classification_model(model_path, num_classes):
                print("Processing image function started")
                model = models.resnet18(pretrained=False)
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, num_classes)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                return model

            def classify_image(image, model, transform, device):
                image = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image)
                    probabilities = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                return preds.item(), probabilities.squeeze().cpu().numpy()

            # загрузка фото
            image = Image.open(image_path)

            # загрузка модели YOLOv5
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)

            # Установка порога уверенности (от 0 до 1)
            model.conf = confidence_threshold

            # Предсказание на изображении
            results = model(image)
            print("YOLO model inference completed")
            print("Results:", results)

            # Загрузка классификационной модели
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            classification_model = load_classification_model(classification_model_path, num_classes).to(device)

            # Подготовка трансформации для классификационной модели
            classification_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Классификация изображения
            class_idx, probabilities = classify_image(image, classification_model, classification_transform, device)
            class_names = ['1-464 A-1', '1-464 A-2']  # Замените на свои классы
            classification_text = f"{class_names[class_idx]}: {probabilities[class_idx]:.2f}"

            # Загрузка шрифта и установка размера текста для детекции
            detection_font_size = int(15 * scale)
            try:
                detection_font = ImageFont.truetype("arial.ttf", detection_font_size)
            except IOError:
                detection_font = ImageFont.load_default()

            # Установка размера текста для классификации
            class_font_size = int(text_size)
            try:
                class_font = ImageFont.truetype("arial.ttf", class_font_size)
            except IOError:
                class_font = ImageFont.load_default()

            # Отображение результатов YOLOv5
            if len(results.xyxy[0]) > 0:
                draw = ImageDraw.Draw(image)
                for detection in results.xyxy[0]:
                    label = int(detection[-1])
                    confidence = detection[-2]
                    box = detection[:-2]
                    box = [(int(coord)) for coord in box]
                    box = tuple(box)
                    if label < len(colors):
                        color = colors[label]
                    else:
                        color = (0, 255, 0)  # default color
                    draw.rectangle(box, outline=color, width=thickness)
                    if show_names:
                        label_name = model.model.names[label]
                        text = f"{label_name}"
                        if show_confidence:
                            text += f" {confidence:.2f}"
                        draw.text((box[0], box[1]), text, fill=color, font=detection_font)

                # Вывод классификационного текста в углу изображения
                if show_probabilities:
                    prob_texts = [f"{class_names[i]}: {prob:.2f}" for i, prob in enumerate(probabilities)]
                    for i, prob_text in enumerate(prob_texts):
                        draw.text((10, 10 + i * (class_font_size + 5)), prob_text, fill=text_color, font=class_font)
                else:
                    draw.text((10, 10), classification_text, fill=text_color, font=class_font)

            return image

        return prediction(image_path, yolo_model_path, classification_model_path, confidence_threshold)


@csrf_exempt
def upload_image_view(request):
    return render(request, 'upload.html')
