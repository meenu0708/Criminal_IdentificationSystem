from django.contrib import admin
from .models import UploadedImage, CriminalImage,Feature, FeatureOption
# Register your models here.

admin.site.register(UploadedImage)
admin.site.register(CriminalImage)
admin.site.register(Feature)
admin.site.register(FeatureOption)