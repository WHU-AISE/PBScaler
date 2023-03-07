import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.concurrent.TimeUnit;


public class TestServiceCellect {
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
    public void testTicketCollect() throws Exception{
        driver.get(baseUrl + "/");
        JavascriptExecutor js = (JavascriptExecutor) driver;
        js.executeScript("document.getElementById('single_collect_order_id').value='5ad7750b-a68b-49c0-a8c0-32776b067703'");
        driver.findElement(By.id("single_collect_button")).click();
        String statusTicketCollect = driver.findElement(By.id("single_collect_order_result")).getText();
        if ("".equals(statusTicketCollect))
            System.out.println("False,status security check is null!");
        else
            System.out.println("Ticket Collect status:"+statusTicketCollect);
        Assert.assertEquals(!"".equals(statusTicketCollect),true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
