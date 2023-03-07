import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;


public class TestServiceExecute {
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
    public void testExecute() throws Exception {
        driver.get(baseUrl + "/");
        driver.findElement(By.id("execute_order_id")).clear();
        driver.findElement(By.id("execute_order_id")).sendKeys("5ad7750b-a68b-49c0-a8c0-32776b067703");
        driver.findElement(By.id("execute_order_button")).click();
        Thread.sleep(1000);
        String statusExecute = driver.findElement(By.id("execute_order_message")).getText();
        if (!"".equals(statusExecute))
            System.out.println("Success: "+statusExecute);
        else
            System.out.println("False, status security check is null!");
        Assert.assertEquals(statusExecute.equals(""),false);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
