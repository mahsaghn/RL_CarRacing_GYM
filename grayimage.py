from PIL import Image

for i in range(1,13):
   img = Image.open(str(i)+'.png').convert('L')
   img.save(str(i)+'.png')

