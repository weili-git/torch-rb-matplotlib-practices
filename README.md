# torch-rb-matplotlib-practices
Study machine learning by torch-rb and matplotlib.

# Environment
- Ubuntu 18.04
- rbenv ruby2.6.6
- torch-rb (require libtorch)
- matplotlib for ruby (require pycall, python, matplotlib)

# Get Started
Get started using the 'Dockerfile'
```bash
sudo apt-get install x11-xserver-utils
xhost +     # enable matplotlib display in docker
docker build -t my-env:1.0 .
docker run -it \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=unix$DISPLAY \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  my-env:1.0
```
if you got trouble when executed `docker build`, just try it again.

# Result
![regression](https://gitee.com/weiligit/codes/3moug12cqkdwyiprejbv897/raw?blob_name=Figure_1.png)
![classification](https://gitee.com/weiligit/codes/16al2zvpdir3o8ksjgymb97/raw?blob_name=Figure_2.png)
![optimizer](https://gitee.com/weiligit/codes/tjvk7mr2fnx01wgbeqyza88/raw?blob_name=Figure_3.png)
![rnn](https://gitee.com/weiligit/codes/t2v4k9ezbrhx8jfd5ni6p20/raw?blob_name=Figure_4.png)
![autoencoder](https://gitee.com/weiligit/codes/bekl6rfyj0cap8w1qzhix76/raw?blob_name=Figure_5.png)
![autoencoder](https://gitee.com/weiligit/codes/bekl6rfyj0cap8w1qzhix76/raw?blob_name=Figure_6.png)
![gan](https://gitee.com/weiligit/codes/cqn4wmvb2s3agxpof6zki35/raw?blob_name=Figure_7.png)

# Reference
https://github.com/MorvanZhou/PyTorch-Tutorial
https://github.com/ankane/torch.rb

