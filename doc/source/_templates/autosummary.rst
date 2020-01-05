{{ fullname | escape | underline}}


.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{% if objtype in ['class', 'method', 'function'] %}

.. raw:: html

    <div class="clear"></div>

{% endif %}
