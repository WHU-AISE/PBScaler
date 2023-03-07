CREATE TABLE `voucherservice`.`voucher` (
  `voucher_id` INT NOT NULL AUTO_INCREMENT,
  `order_id` VARCHAR(1024) NOT NULL,
  `travelDate` DATE NOT NULL,
  `travelTime` VARCHAR(1024) NOT NULL,
  `contactName` VARCHAR(1024) NOT NULL,
  `trainNumber` VARCHAR(1024) NOT NULL,
  `seatClass` INT NOT NULL,
  `seatNumber` VARCHAR(1024) NOT NULL,
  `startStation` VARCHAR(1024) NOT NULL,
  `destStation` VARCHAR(1024) NOT NULL,
  `price` FLOAT NOT NULL,
  PRIMARY KEY (`voucher_id`));