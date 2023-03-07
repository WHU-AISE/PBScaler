import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.Select;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.concurrent.TimeUnit;

public class TestServiceNotification {
    private WebDriver driver;
    private String baseUrl;
    @BeforeClass
    public void setUp() throws Exception {
        System.setProperty("webdriver.chrome.driver", "D:/Program/chromedriver_win32/chromedriver.exe");
        driver = new ChromeDriver();
        baseUrl = "http://10.141.212.24/";
        driver.manage().timeouts().implicitlyWait(30, TimeUnit.SECONDS);
    }
    @Test
    public void testNotification() throws Exception{
        driver.get(baseUrl + "/");
        JavascriptExecutor js = (JavascriptExecutor) driver;
        js.executeScript("document.getElementById('notification_email').value='daihongok@163.com'");
        js.executeScript("document.getElementById('notification_orderNumber').value='123456789'");
        js.executeScript("document.getElementById('notification_username').value='fdse_microservices'");
        js.executeScript("document.getElementById('notification_startingPlace').value='Shang Hai'");
        js.executeScript("document.getElementById('notification_endPlace').value='Tai Yuan'");

        String jsstartingTime = "document.getElementById('notification_startingTime').value='11:55'";
        js.executeScript(jsstartingTime);
        String jssendTime = "document.getElementById('notification_date').value='2017-8-8'";
        js.executeScript(jssendTime);

        js.executeScript("document.getElementById('ticketinfo_startingPlace').value='Shang Hai'");
        js.executeScript("document.getElementById('ticketinfo_endPlace').value='Tai Yuan'");

        js.executeScript("document.getElementById('notification_seatClass').value='economyClass'");
        js.executeScript("document.getElementById('notification_seatNumber').value='2'");
        js.executeScript("document.getElementById('notification_price').value='1000'");

        WebElement elementNotificationtype = driver.findElement(By.id("notification_type"));
        Select selNotifType = new Select(elementNotificationtype);
        selNotifType.selectByValue("0"); //Preserve Success
        driver.findElement(By.id("notification_send_email_button")).click();
        Thread.sleep(1000);

        //get Notification status
        String statusSendemail = driver.findElement(By.id("notification_result")).getText();
        if("".equals(statusSendemail))
            System.out.println("Failed to Send email! Send email status is NULL");
        else if(statusSendemail.startsWith("true"))
            System.out.println("Send email status:"+statusSendemail);
        else
            System.out.println("Failed to Send email! Send email status："+statusSendemail);
        Assert.assertEquals(statusSendemail.startsWith("true"),true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
