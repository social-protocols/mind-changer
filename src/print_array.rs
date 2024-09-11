use ndarray::Array2;

pub fn print_array(array: &Array2<f64>) {
    for row in array.outer_iter() {
        for &value in row.iter() {
            // Clamp the value between -1.0 and 1.0
            let value = value.clamp(-1.0, 1.0);

            // Calculate the RGB values based on the value
            let (r, g, b) = if value < 0.0 {
                // Fade from yellow (255, 255, 0) to black (0, 0, 0)
                let level = (255.0 * (-value)).round() as u8;
                (level, level, 0)
            } else {
                // Fade from black (0, 0, 0) to blue (0, 0, 255)
                let level = (255.0 * value).round() as u8;
                (0, 0, level)
            };

            // Print the block with the appropriate color
            print!("\x1b[48;2;{};{};{}m  \x1b[0m", r, g, b);
        }
        println!();
    }
}
