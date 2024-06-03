from django.urls import path
from .views import MlModelAPIView, upload_image_view

urlpatterns = [
    path('api/ml-model/', MlModelAPIView.as_view(), name='ml_model_api'),
    path('upload/', upload_image_view, name='upload_image'),
]
