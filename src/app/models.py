from django.db import models

# Create your models here.

class Record(models.Model):
    model_name = models.CharField(max_length=100)
    uploader = models.CharField(max_length=100)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()
    def __str__(self):
        return self.model_name
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Record'
        verbose_name_plural = 'Records'
    