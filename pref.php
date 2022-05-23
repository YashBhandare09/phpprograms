<?php 
  $fn=$_POST['fname'];
  $fc=$_POST['fcolor'];
  $bc=$_POST['bcolor'];
  $size=$_POST['size'];

 setcookie('fn',$fn);
 setcookie('fc',$fc);
 setcookie('bc',$bc);
 setcookie('size',$size);
?>
<h3>your preferences:</h3>
Font Name:<?=$fn?><br>
Fore color:<?=$fc?><br>
Back color:<?=$bc?><br>
size:<?=size?><br>
click <a href='pref1.php'>here</a>to see settings







