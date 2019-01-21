#git使用
1.安装git：sudo apt-get install git  
2.创建github账号，包括用户名，邮箱  
3. 创建ssh：ssh-keygen -t rsa -C "18829354152@163.com"  
4.在自己的github页面找到SSH Keys设置，titile随便填，将本地用户隐藏文件夹C:\Users\Apep\.ssh里的字符复制到add ssh key里。  
5.测试ssh是否成功：ssh -T git@github.com  
6.配置git的配置文件（用户名和邮箱一定要与自己注册的github账号用户名和邮箱相同）：  
git config --global user.name "your name"   
git config --global user.email "your email"   
7.在github上新建一个repository  
8.利用git从本地上传到github  
-8.1进入所要上传文件的目录输入：git init 目的是把当前目录变成git可以管理的仓库  
-8.2创建本地仓库：git remote add origin git@github.com:yourName/yourRepo.git   
上传文件（3条指令）：  
单个文件git add xxx.txt 全部文件git add -A  
git commit -m "提交信息"  
第一次提交git push -u origin master 以后提交git push origin master  
-8.3查看是否还有未提交:git status  
查看最近日志:git log  
-8.4版本回退:回退一个:git reset -hard HEAD^  
回退两个:git reset -hard HEAD^^  
回退多个:git reset -hard HEAD~100(100即回退次数)  
9.从github克隆项目到本地  
在本地建立接受代码的文件夹，cd到这个目录，所使用的远程主机自动被个i他命名为origin  
如果想用其他主机名，需要用个git clone命令的-o选项指定。  
代码$git clone -o github https://github.com/xxxxxx(仓库地址) 注：该代码已将远程主机名改为github  
10.github创建新分支  
在本地创建新的分支 git branch newbranch  
切换到新的分支 git checkout newbranch  
将新的分支推送到github git push origin newbranch  
在本地删除一个分支： git branch -d newbranch  
在github远程端删除一个分支： git push origin :newbranch (分支名前的冒号代表删除)  
复制仓库的Https，使用命令git clone https://github.com/PentonBin/Demo.git（例子）  
11.查看远程主机名和fetch push地址：git remote -v  
12.git clone和git pull区别：  
git clone是将整个版本库复制到本地，git pull=git fetch + git merge即从远程主机获取最新版本branch分支并合并到本地。   
13.使用git命令每次提交都要输入用户名和密码的解决办法  
1)git remote rm origin   
2)git remote add origin https://username:password@github.com/username/mlnotes/  
3)git push origin master  

#分支的增删查改
1.查看分支：git branch  
2.查看远程所有分支：git branch -r  
3.查看本地和远程所有分支：git branch -a  
4.创建分支：git branch <`name`>  
5.切换分支：git checkout <`name`>  
6.创建并切换分支：git checkout -b <`name`>  
7.合并某分支到当前分支：git merge <`name`>  
8.把分支推送到远程：git push origin <`name`>  
9.删除本地分支：git branch -d <`name`>  不能删除当前所在的本地分支  
10.删除远程分支：git push origin -d <`name`>  
11.分支重命名：git branch -m <`old`> <`new`>  
12.合并某个分支： git merge <`name`>  
13.对比两个分支： git diff <`name1`> <`name2`>  
14.对比本地和远程分支：git diff <`name`> origin/<`name`>  
15.强制覆盖本地分支：  
（1）git fetch -all  
（2）git reset --hard origin/<`name`>  
（3）git pull 
#查看提交信息日志
1.查看分支最近一次的修改列表：git status  
2.查看分支的commit信息：  
（1）`git log` 查看commit id,Author,Date,commit info  
（2）`git shortlog` 按提交者分类显示提交信息  
（3）`git log --oneline` 只输出commit id和commit info  
（4）`git log --stat` 查看增删查改了哪些文件  

#版本回退  
1.回退到上一个版本： git reset --hard HEAD^  
2.回退到上上个版本： git reset --hard HEAD^^  
3.回退到上n个版本： git reset --hard HEAD~n  
4.回退到某个版本： git reset --hard <`commit id`>  

#常用选项和其他命令
1.`-f` --force:强制  
2.`-d` --delete:删除  
3.`-m` --move:移动或重命名  
4.`-r` -remote:远程  
5.`-a` -all :所有  

#清空工程
git rm -rf .