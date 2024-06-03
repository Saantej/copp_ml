from django.contrib import admin


from . import models


@admin.register(models.UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    pass