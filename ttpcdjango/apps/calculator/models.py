from django.db import models

# Create your models here.

class QueryLog(models.Model):
    query = models.CharField(max_length=255)
    desc = models.TextField(
        help_text="Description for Query.",
        blank=True, null=True)
    viewed = models.BooleanField(
        default=True)
    ip_address = models.GenericIPAddressField(
        blank=True, null=True)

    created = models.DateTimeField(
        auto_now_add=True)
    modified = models.DateTimeField(
        auto_now=True)

    class Meta:
        verbose_name = "Query"
        verbose_name_plural = "Queries"
        ordering = ["-created"]

    def _str_(self):
        return "{} {}".format(
            self.id, self.query)