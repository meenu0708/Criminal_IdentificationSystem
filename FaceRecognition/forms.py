from django import forms
from .models import UploadedImage

class ImageForm(forms.ModelForm):
    class Meta:
        model=UploadedImage
        fields=['image']
        error_messages = {
            'image': {
                'required': '',
            }
        }

class CanvasImageForm(forms.Form):
    canvas_image_data = forms.CharField(widget=forms.HiddenInput())
    image_name = forms.CharField(max_length=50)

class LoginForm(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
    )

class UserForm(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
    )
    cpass = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
    )