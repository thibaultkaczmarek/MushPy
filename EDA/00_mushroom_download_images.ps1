

$step = 25000

for ($i=0; $i -le 700000; $i+=$step ) {
    write-host($i);

    $min_index = $i
    $max_index = $i + $step
    Start-Process python.exe -ArgumentList "-file C:\Projets\MushroomRecognition\mushroom_download_images.ps1 $min_index, $max_index"  
    
}


#Start-Process python.exe -ArgumentList "-file C:\Projets\MushroomRecognition\mushroom_download_images.ps1", 1, 2  


