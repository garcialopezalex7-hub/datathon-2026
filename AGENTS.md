# Reglas de flujo de trabajo con GitHub

Este proyecto está conectado a un repositorio de GitHub compartido en equipo. Sigue siempre estas reglas:

1. **Antes de empezar cualquier tarea o editar archivos:**
   - Ejecuta `git status` y `git pull --rebase` (o `git pull`) para asegurarte de tener los últimos cambios del repositorio y evitar conflictos.
   - Si hay cambios locales sin guardar o conflictos pendientes, avisa antes de modificar archivos.

2. **Al terminar cada cambio o funcionalidad importante (una vez verificado que funciona):**
   - Revisa `git status` y `git diff --stat` para comprobar que no se incluyan archivos ignorados, temporales, `.venv/`, `__pycache__/` ni datasets pesados.
   - Haz `git add` de los archivos modificados, crea un `git commit -m "..."` con un mensaje claro y descriptivo de lo que se ha hecho, y súbelo con `git push`.

3. **Seguridad:**
   - Nunca uses `git push --force` ni borres historial de commits.
   - Si falla el `git push` porque hay cambios nuevos en remoto, haz `git pull --rebase` primero; si aparecen conflictos, detente y pregunta cómo resolverlos.
