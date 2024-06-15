from django.db import models
from django.utils import timezone
# Create your models here.
class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')

    class Meta:
        pass

class CriminalImage(models.Model):
    image_id = models.AutoField(primary_key=True)
    sketched_image = models.ImageField(upload_to='sketched_images/')
    original_image = models.ImageField(upload_to='original_images/')
    created_at = models.DateTimeField(default=timezone.now)

class Feature(models.Model):
    name = models.CharField(max_length=250, unique=True)
    slug = models.SlugField(max_length=250, unique=True)

    class Meta:
        ordering=('name',)
        verbose_name='feature'
        verbose_name_plural='features'

    def __str__(self):
        return '{}'.format(self.name)

class FeatureOption(models.Model):
    feature = models.ForeignKey(Feature, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='feature_options/')

