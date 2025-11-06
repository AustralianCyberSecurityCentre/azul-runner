The Azul framework allows plugins to mark their output with further security restrictions. 

This can be useful when the plugin implements a sensitive algorithm or does data enrichment from a restricted source. 

```python
# Plaintext security applied to all results of a plugin.
# Examples:
# SECURITY = "UNCLASSIFIED"
# SECURITY = "UNCLASSIFIED TLP:GREEN"
# SECURITY = "AMAZING TOMATO TLP:GREEN REL:ME,YOU,THATONE"
SECURITY: ClassVar[Optional[str]] = None
```

For example, if your install defines the groups below and you require users to be in these groups
to see your plugin's output:
```python
class MySensitivePlugin(BinaryPlugin):
    SECURITY = "SOC_ONLY VICTIM_DETAILS"
```

Restrictions set by a plugin will be combined with restrictions on original event.

If your plugin will be deployed with sensitive runtime info, 
you may not be able to hard code the security level required.

Set the `security_override` config option to override the default security string the plugin specifies.
