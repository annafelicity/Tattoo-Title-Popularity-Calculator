from django.contrib import admin
from .models import QueryLog

# Register your models here.

class QueryLogAdmin(admin.ModelAdmin):
    readonly_fields = ['created', 'modified', 'ip_address']
    list_display = ['id', 'query', 'created', 'modified']
    search_fields = ['query']


admin.site.register(QueryLog, QueryLogAdmin)