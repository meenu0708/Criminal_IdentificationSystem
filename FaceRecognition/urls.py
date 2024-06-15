from django.urls import path
from .import views
urlpatterns = [

    path('', views.home, name='home'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('upload_sketch',views.upload_sketch,name='upload_sketch'),
    path('upload_sketch/sketch_match/', views.sketch_match, name='sketch_match'),
    path('sketch_interface', views.sketch_interface, name='sketch_interface'),
    path('features/<slug:feature_slug>/', views.feature_detail, name='feature_detail'),
    path('save_image/', views.save_image, name='save_image'),
]
