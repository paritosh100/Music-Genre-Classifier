from django.urls import path
from . import views

app_name = 'classifier_app' # Namespace for your app's URLs

urlpatterns = [
    path('', views.upload_file_and_predict, name='upload_predict'),
]
