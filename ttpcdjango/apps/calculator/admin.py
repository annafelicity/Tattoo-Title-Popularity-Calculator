from django.contrib import admin
from .models import QueryLog

# Register your models here.

class QueryLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'query']
    search_fields = ['query']


admin.site.register(QueryLog, QueryLogAdmin)