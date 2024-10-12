# Proyecto1-Etapa 2


## Deployment

> **Nota:** Este proyecto debe abrirse utilizando Anaconda. Asegúrate de tener Anaconda instalado y activa el entorno correspondiente antes de ejecutar los comandos mencionados.


1.  **Instalación de librerías**: Antes de poner en marcha la API, es necesario instalar las librerías requeridas. Puedes hacerlo ejecutando la siguiente línea en tu consola:

```bash
pip install fastapi
pip install "uvicorn[standard]" 
```
2. Luego, Accede a la carpeta del proyecto con el siguiente comando:
```bash
cd aplicacion
```

3. Inicie el proyecto con el siguiente comando:

```bash
python -m uvicorn main:app --reload
```

4. Una vez que la API esté en funcionamiento, abre tu navegador y dirígete a la siguiente URL:
```bash
http://127.0.0.1:8000
```

Si se ejecuto correctamente se debe ver de la siguiente forma:

![image](https://github.com/user-attachments/assets/ba1be8dc-b70d-4d31-b32d-6eee170c9978)

5. Si deseas detener la ejecución de la API, simplemente presiona las teclas Ctrl + C en la consola donde la API se está ejecutando.
